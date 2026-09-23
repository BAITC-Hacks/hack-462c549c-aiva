import re
from difflib import SequenceMatcher

from .models import ComparisonRow, FunctionRecord, ParsedDocument


def similarity(left: str, right: str) -> float:
    return round(SequenceMatcher(None, normalize(left), normalize(right)).ratio(), 3)


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    return re.sub(r"[^\wа-я ]", " ", text, flags=re.I).replace("  ", " ").strip()


def compare_documents(before: ParsedDocument, after: ParsedDocument) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    all_clause_ids = sorted(set(before.clauses) | set(after.clauses), key=_clause_sort_key)
    for clause_id in all_clause_ids:
        old = before.clauses.get(clause_id)
        new = after.clauses.get(clause_id)
        if old and new:
            score = similarity(old.cleaned_text, new.cleaned_text)
            if score >= 0.995:
                status, kind, explanation = "Сохранена", "FACT", "Пункт присутствует в обеих редакциях без существенного текстового отличия."
            else:
                status, kind, explanation = classify_changed_clause(clause_id, old.cleaned_text, new.cleaned_text)
            rows.append(ComparisonRow(
                before=FunctionRecord(function_id=old.fragment_id, text=old.cleaned_text, fragment_id=old.fragment_id, clause_id=old.clause_id),
                after=FunctionRecord(function_id=new.fragment_id, text=new.cleaned_text, fragment_id=new.fragment_id, clause_id=new.clause_id),
                status=status, evidence_type=kind, explanation=explanation, confidence=score,
            ))
        elif old:
            rows.append(ComparisonRow(
                before=FunctionRecord(function_id=old.fragment_id, text=old.cleaned_text, fragment_id=old.fragment_id, clause_id=old.clause_id),
                status="Удалена / не найдена в ПОСЛЕ", evidence_type="RISK_FLAG",
                explanation="Пункт есть в документе ДО, но пункт с таким идентификатором не найден в ПОСЛЕ; требуется проверка.", confidence=1.0,
            ))
        else:
            rows.append(ComparisonRow(
                after=FunctionRecord(function_id=new.fragment_id, text=new.cleaned_text, fragment_id=new.fragment_id, clause_id=new.clause_id),
                status="Новая", evidence_type="FACT", explanation="Пункт присутствует только в документе ПОСЛЕ.", confidence=1.0,
            ))
    return rows


def classify_changed_clause(clause_id: str, old: str, new: str):
    if clause_id.startswith("3."):
        return "Изменена", "INFERENCE", "Изменение относится к структуре, подчинённости или составу должностей."
    if clause_id.startswith("2.") or clause_id.startswith("4.") or clause_id.startswith("5.") or clause_id.startswith("9."):
        return "Изменена", "INFERENCE", "Изменён текст пункта, связанного с функциями или процедурами; функциональный эффект требует проверки."
    return "Изменена", "FACT", "Текст пункта отличается между редакциями."


def _clause_sort_key(value: str):
    return tuple(int(part) if part.isdigit() else 999 for part in value.split("."))


def structural_changes(before: ParsedDocument, after: ParsedDocument) -> list[dict]:
    old = {item["unit_name"] for item in before.units}
    new = {item["unit_name"] for item in after.units}
    return ([{"name": n, "status": "Новая"} for n in sorted(new - old)] +
            [{"name": n, "status": "Сохранена"} for n in sorted(old & new)] +
            [{"name": n, "status": "Удалена / не найдена в ПОСЛЕ"} for n in sorted(old - new)])
