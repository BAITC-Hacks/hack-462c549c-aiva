import pandas as pd

from .models import ComparisonRow, ParsedDocument


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
