import pandas as pd

from .models import ComparisonRow, ParsedDocument


def review_key(row: ComparisonRow) -> str:
    source = row.before or row.after
    return source.fragment_id if source else row.status


def review_priority(row: ComparisonRow) -> dict:
    status = row.semantic_status or row.status
    if row.result_type == "RISK_FLAG" or status in {"потенциально потеряна", "потенциально перераспределена", "потенциальное дублирование", "потенциальный конфликт полномочий"}:
        return {"priority": "Проверить в первую очередь", "reason": "RISK_FLAG или потенциальное изменение ответственности."}
    if row.requires_human_review or status == "функция существенно изменена":
        return {"priority": "Требует проверки", "reason": "Неоднозначное или существенно изменённое сопоставление."}
    return {"priority": "Информационно", "reason": "Результат не содержит признаков обязательной экспертной проверки."}


def expert_conclusion(rows: list[ComparisonRow], review_state: dict) -> list[dict]:
    result = []
    for row in rows:
        state = review_state.get(review_key(row), {})
        if state.get("decision") == "Подтверждено":
            result.append({"status": "Подтверждено экспертом", "text": row.explanation, "source": review_key(row)})
        elif state.get("decision") == "Отклонено":
            result.append({"status": "Отклонено экспертом", "text": "AI-находка отклонена и не считается подтверждённой.", "source": review_key(row)})
        elif row.requires_human_review or row.result_type == "RISK_FLAG":
            result.append({"status": "Ожидает экспертной проверки", "text": row.explanation, "source": review_key(row)})
    return result


def build_recommendations(rows: list[ComparisonRow]) -> list[dict]:
    recommendations = []
    for row in rows:
        source = row.before or row.after
        if not source:
            continue
        if row.semantic_status in {"потенциально перераспределена", "потенциально потеряна"}:
            recommendations.append({"priority": "требует проверки", "observation": row.semantic_status,
                "action": "Проверить распределение и закрепление функции.", "source": source.fragment_id,
                "confidence": row.confidence})
        elif row.requires_human_review and (row.before and row.after and row.before.clause_id and row.before.clause_id.startswith("3.")):
            recommendations.append({"priority": "требует проверки", "observation": "Изменение структуры или ответственности",
                "action": "Проверить корректность закрепления ответственности за подразделением.", "source": source.fragment_id,
                "confidence": row.confidence})
        elif row.semantic_status in {"потенциальное дублирование", "потенциальный конфликт полномочий"}:
            recommendations.append({"priority": "требует проверки", "observation": row.semantic_status,
                "action": "Проверить отсутствие пересечения полномочий.", "source": source.fragment_id,
                "confidence": row.confidence})
    return recommendations


def rows_to_dataframe(rows: list[ComparisonRow]) -> pd.DataFrame:
    data = []
    for row in rows:
        old = row.before
        new = row.after
        data.append({
            "Пункт ДО": old.clause_id if old else "",
            "Страница ДО": _page(old, rows),
            "Фрагмент ДО": old.text if old else "",
            "Пункт ПОСЛЕ": new.clause_id if new else "",
            "Страница ПОСЛЕ": _page(new, rows),
            "Фрагмент ПОСЛЕ": new.text if new else "",
            "Статус": row.status,
            "Тип": row.evidence_type,
            "Объяснение": row.explanation,
            "Confidence": row.confidence,
        })
    return pd.DataFrame(data)


def _page(record, rows):
    # Page is presented by the UI from the fragment map; this function keeps the table schema stable.
    return ""


def conclusion(before: ParsedDocument, after: ParsedDocument, rows: list[ComparisonRow], ai_available: bool, structure=None, duplicates=None) -> str:
    counts = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    semantic_counts = {}
    for row in rows:
        if row.semantic_status:
            semantic_counts[row.semantic_status] = semantic_counts.get(row.semantic_status, 0) + 1
    lines = [
        f"Сравнены документы: {before.document_name} и {after.document_name}.",
        f"Извлечено фрагментов: ДО — {len(before.fragments)}, ПОСЛЕ — {len(after.fragments)}.",
        f"Подразделений в ДО: {len(before.units)}, в ПОСЛЕ: {len(after.units)}.",
        "Изменения классифицированы по структурным пунктам и semantic matching функций.",
    ]
    if structure:
        lines.append("\nСтруктурные изменения:")
        for item in structure:
            lines.append(f"- {item['status']}: {item['name']}")
    lines.append("\nФункциональные изменения:")
    for status in ("смысл сохранен", "редакционное изменение", "функция уточнена", "функция расширена", "функция сокращена", "функция существенно изменена", "потенциально перераспределена", "потенциально потеряна"):
        lines.append(f"- {status}: {semantic_counts.get(status, 0)}")
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")
    risks = [r for r in rows if r.result_type == "RISK_FLAG"] + list(duplicates or [])
    if risks:
        lines.append("\nПотенциальные риски:")
        for row in risks[:20]:
            source = row.before or row.after
            lines.append(f"- {row.status}: {row.explanation} [{source.fragment_id if source else 'source unavailable'}]")
    review = [r for r in rows if r.requires_human_review]
    lines.append(f"\nТребуют проверки человеком: {len(review)} результатов.")
    for row in review[:20]:
        source = row.before or row.after
        lines.append(f"- {row.semantic_status or row.status}: {source.fragment_id if source else 'source unavailable'}")
    if not ai_available:
        lines.append("Semantic AI недоступен: использован deterministic fallback без внешнего API.")
    return "\n".join(lines)
