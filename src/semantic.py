"""Bounded semantic analysis layered on top of the deterministic comparison."""

import json
import os
import re
from difflib import SequenceMatcher

from .analyzer import normalize
from .models import ComparisonRow, Fragment, FunctionRecord, ParsedDocument

SEMANTIC_STATUSES = {
    "смысл сохранен", "редакционное изменение", "функция уточнена", "функция расширена",
    "функция сокращена", "функция существенно изменена", "потенциально перераспределена",
    "потенциально потеряна", "недостаточно данных / требуется проверка человеком",
}
RISK_STATUSES = {"потенциально потеряна", "потенциально дублируется", "потенциальный конфликт полномочий"}
REVIEW_THRESHOLD = 0.70
CONSISTENCY_THRESHOLD = 0.85
LAST_AI_ERROR: dict | None = None


def model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _error_info(exc: Exception) -> dict:
    name = type(exc).__name__
    text = str(exc).lower()
    if "authentication" in text or "api key" in text or "401" in text:
        category = "authentication error"
    elif "rate limit" in text or "429" in text or "quota" in text:
        category = "quota/rate limit"
    elif "model" in text and ("not found" in text or "does not exist" in text or "404" in text):
        category = "model unavailable"
    elif "json" in text or "parse" in text or "structured" in text:
        category = "parsing/structured response error"
    elif "timeout" in text or "connection" in text or "network" in text:
        category = "network error"
    else:
        category = "SDK error"
    raw = str(exc)
    if category == "authentication error":
        safe_message = "Authentication error: API key rejected by OpenAI."
    else:
        safe_message = _sanitize_message(raw)
    return {"category": category, "exception": name, "message": safe_message}


def _sanitize_message(message: str) -> str:
    """Return diagnostics safe for UI display; never expose credentials."""
    safe = message
    configured_key = os.getenv("OPENAI_API_KEY")
    if configured_key:
        safe = safe.replace(configured_key, "[REDACTED]")
    safe = re.sub(r"(?i)(api[_ -]?key|authorization|token|secret|credential)(\s*[:=]\s*)([^\s,;]+)", r"\1\2[REDACTED]", safe)
    safe = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", safe)
    safe = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", safe)
    return safe[:500]


def ai_status() -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "Not configured", "model": model_name(), "error": None}
    if LAST_AI_ERROR:
        return {"status": "API error / fallback", "model": model_name(), "error": LAST_AI_ERROR}
    return {"status": "Configured", "model": model_name(), "error": None}


def candidate_pairs(before: ParsedDocument, after: ParsedDocument, rows: list[ComparisonRow], limit: int = 3):
    """Find small candidate sets; no whole-document prompt is ever built."""
    old = [r.before for r in rows if r.before and not r.after]
    new = [r.after for r in rows if r.after and not r.before]
    changed = [(r.before, r.after) for r in rows if r.before and r.after and r.status != "Сохранена"]
    pairs = []
    for item in old:
        ranked = sorted(((similarity(item.text, candidate.text), candidate) for candidate in new), reverse=True, key=lambda x: x[0])
        pairs.extend((item, candidate, score) for score, candidate in ranked[:limit] if score >= 0.18)
    pairs.extend((left, right, similarity(left.text, right.text)) for left, right in changed)
    return pairs


def similarity(left: str, right: str) -> float:
    return round(SequenceMatcher(None, normalize(left), normalize(right)).ratio(), 3)


def _fallback(before: FunctionRecord, after: FunctionRecord, score: float) -> dict:
    if score >= 0.70:
        status = "смысл сохранен"
    elif score >= 0.42:
        status = "функция уточнена"
    else:
        status = "недостаточно данных / требуется проверка человеком"
    return {
        "before_fragment_id": before.fragment_id,
        "after_fragment_ids": [after.fragment_id],
        "semantic_status": status,
        "confidence": score,
        "explanation": "Deterministic fallback: кандидат найден по текстовой близости; semantic AI недоступен.",
        "requires_human_review": score < 0.70,
        "result_type": "FACT" if score >= 0.70 else "INFERENCE",
    }


def call_ai(before: FunctionRecord, after: FunctionRecord, candidates: list[FunctionRecord] | None = None) -> dict | None:
    global LAST_AI_ERROR
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        model = model_name()
        payload = {
            "before": {"fragment_id": before.fragment_id, "clause_id": before.clause_id, "text": before.text},
            "after_candidates": [{"fragment_id": x.fragment_id, "clause_id": x.clause_id, "text": x.text} for x in (candidates or [after])],
        }
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Analyze only the supplied function fragments. Return JSON with before_fragment_id, after_fragment_ids, semantic_status, confidence, explanation, requires_human_review. Use only supplied fragment IDs. Focus on actor, action, object, scope. Never invent sources."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("empty structured response")
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("structured response is not a JSON object")
        LAST_AI_ERROR = None
        return result
    except Exception as exc:
        LAST_AI_ERROR = _error_info(exc)
        return None


def check_ai_connection(before: FunctionRecord, after: FunctionRecord) -> dict:
    """One-pair smoke check; it never sends the whole document."""
    global LAST_AI_ERROR
    if not os.getenv("OPENAI_API_KEY"):
        return ai_status()
    result = call_ai(before, after)
    if result is None:
        return ai_status()
    try:
        validated = validate_result(result, {before.fragment_id}, {after.fragment_id}, before.fragment_id, after.fragment_id)
        return {"status": "Connected", "model": model_name(), "result": validated, "error": None}
    except Exception as exc:
        LAST_AI_ERROR = _error_info(exc)
        return ai_status()


def validate_result(result: dict, before_ids: set[str], after_ids: set[str], expected_before: str, expected_after: str) -> dict:
    after_ids_result = result.get("after_fragment_ids") or []
    if isinstance(after_ids_result, str):
        after_ids_result = [after_ids_result]
    if result.get("before_fragment_id") != expected_before or result.get("before_fragment_id") not in before_ids:
        raise ValueError("AI returned an invalid before_fragment_id")
    if not after_ids_result or any(x not in after_ids for x in after_ids_result):
        raise ValueError("AI returned an invalid after_fragment_id")
    explanation = str(result.get("explanation", ""))
    status = str(result.get("semantic_status", "")).lower().strip()
    lower_explanation = explanation.lower()
    # Resolve direct contradictions from the model before assigning result_type.
    if any(word in lower_explanation for word in ("identical", "no change", "same meaning", "без изменений", "смысл сохранен")):
        status = "смысл сохранен"
    elif any(word in lower_explanation for word in ("expand", "расшир", "добав")) and status in {"недостаточно данных / требуется проверка человеком", "потенциально потеряна"}:
        status = "функция расширена"
    elif any(word in lower_explanation for word in ("clarif", "уточн")) and status in {"недостаточно данных / требуется проверка человеком", "потенциально потеряна"}:
        status = "функция уточнена"
    if status not in SEMANTIC_STATUSES:
        if any(word in lower_explanation for word in ("identical", "no change", "same meaning", "без изменений", "смысл сохранен")):
            status = "смысл сохранен"
        elif any(word in lower_explanation for word in ("expand", "расшир", "добав")):
            status = "функция расширена"
        elif any(word in lower_explanation for word in ("clarif", "уточн")):
            status = "функция уточнена"
        else:
            status = "недостаточно данных / требуется проверка человеком"
    confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    consistency_text = lower_explanation
    actor_or_scope_changed = any(term in consistency_text for term in (
        "actor changed", "different actor", "responsibility moved", "subdivision changed",
        "department changed", "scope substantially changed", "transferred", "перешла",
        "подразделение измен", "ответственность измен", "существенно изменен scope",
    ))
    explicit_maintained = any(term in consistency_text for term in (
        "meaning maintained", "semantically consistent", "no meaningful change",
        "same action/responsibility", "same action", "action maintained",
        "meaning is preserved", "смысл сохранен",
    ))
    explicit_clarified = any(term in consistency_text for term in (
        "clarified", "detailed", "expanded", "уточнен", "детализирован", "расширен",
    ))
    if confidence >= CONSISTENCY_THRESHOLD and not actor_or_scope_changed:
        if explicit_maintained:
            status = "редакционное изменение" if any(term in consistency_text for term in ("editorial", "wording", "редакцион")) else "смысл сохранен"
            result["result_type"] = "FACT"
            result["requires_human_review"] = False
        elif explicit_clarified:
            status = "функция расширена" if any(term in consistency_text for term in ("expanded", "расширен")) else "функция уточнена"
            result["result_type"] = "FACT"
            result["requires_human_review"] = False
    if status == "смысл сохранен" and confidence >= REVIEW_THRESHOLD:
        result_type = "FACT"
    elif status in {"редакционное изменение", "функция уточнена", "функция расширена"} and confidence >= REVIEW_THRESHOLD:
        result_type = "FACT"
    elif status in {"функция сокращена", "функция существенно изменена", "потенциально перераспределена"}:
        result_type = "INFERENCE"
    elif status in RISK_STATUSES:
        result_type = "RISK_FLAG"
    else:
        result_type = "INFERENCE"
    requires_review = (
        status == "недостаточно данных / требуется проверка человеком" or
        confidence < REVIEW_THRESHOLD or
        status in {"потенциально потеряна", "потенциально перераспределена", "потенциально дублируется", "потенциальный конфликт полномочий"}
    )
    return {**result, "after_fragment_ids": after_ids_result, "semantic_status": status,
            "confidence": confidence, "requires_human_review": requires_review,
            "result_type": result_type}


def enhance_rows(before: ParsedDocument, after: ParsedDocument, rows: list[ComparisonRow], use_ai: bool = True):
    before_ids = {f.fragment_id for f in before.fragments}
    after_ids = {f.fragment_id for f in after.fragments}
    pairs = candidate_pairs(before, after, rows)
    enhanced = list(rows)
    matched_before, matched_after = set(), set()
    for old, new, score in pairs:
        result = call_ai(old, new) if use_ai else None
        method = "AI" if result else "deterministic"
        result = result or _fallback(old, new, score)
        try:
            result = validate_result(result, before_ids, after_ids, old.fragment_id, new.fragment_id)
        except (TypeError, ValueError):
            result = _fallback(old, new, score)
            method = "deterministic"
        for row in enhanced:
            if row.before and row.before.fragment_id == old.fragment_id and (not row.after or row.after.fragment_id == new.fragment_id):
                row.semantic_status = result["semantic_status"]
                row.confidence = result["confidence"]
                row.explanation = result["explanation"]
                row.requires_human_review = result["requires_human_review"]
                row.analysis_method = method
                row.result_type = result["result_type"]
                row.evidence_type = result["result_type"]
                if result["semantic_status"] == "потенциально перераспределена":
                    row.status = "Потенциально перераспределена"
                elif result["semantic_status"] == "потенциально потеряна":
                    row.status = "Потенциально потеряна"
                elif row.status == "Удалена / не найдена в ПОСЛЕ" and result["semantic_status"] != "недостаточно данных / требуется проверка человеком":
                    row.status = "Изменена"
                matched_before.add(old.fragment_id); matched_after.add(new.fragment_id)
                break
    # A missing clause is not itself a risk. Only unresolved function candidates remain a review item.
    for row in enhanced:
        if row.status == "Удалена / не найдена в ПОСЛЕ":
            row.evidence_type = "INFERENCE"
            row.result_type = "INFERENCE"
            row.semantic_status = "недостаточно данных / требуется проверка человеком"
            row.requires_human_review = True
            row.analysis_method = "deterministic"
            row.explanation = "Пункт не найден по clause_id; это не доказывает потерю функции. Требуется semantic review."
    return enhanced, len(pairs)


def duplicate_flags(after: ParsedDocument, use_ai: bool = True) -> list[ComparisonRow]:
    """Find only cross-unit duplicates when unit context is available."""
    result = []
    functions = [x for x in after.functions if x.unit_name]
    for index, left in enumerate(functions):
        for right in functions[index + 1:]:
            if left.unit_name == right.unit_name:
                continue
            score = similarity(left.text, right.text)
            if score >= 0.78:
                result.append(ComparisonRow(
                    before=None, after=left, status="Потенциальное дублирование", evidence_type="RISK_FLAG",
                    semantic_status="потенциальное дублирование", confidence=score,
                    explanation=f"Функции подразделений '{left.unit_name}' и '{right.unit_name}' семантически близки; требуется проверка.",
                    requires_human_review=True, analysis_method="deterministic", result_type="RISK_FLAG"))
    return result
