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


def conclusion(before: ParsedDocument, after: ParsedDocument, rows: list[ComparisonRow], ai_available: bool) -> str:
    counts = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    lines = [
        f"Сравнены документы: {before.document_name} и {after.document_name}.",
        f"Извлечено фрагментов: ДО — {len(before.fragments)}, ПОСЛЕ — {len(after.fragments)}.",
        f"Подразделений в ДО: {len(before.units)}, в ПОСЛЕ: {len(after.units)}.",
        "Изменения классифицированы по номерам структурных пунктов и текстовому сходству.",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")
    lines.append("Потенциальные риски и спорные выводы требуют проверки человеком.")
    if not ai_available:
        lines.append("Semantic AI недоступен: использован deterministic fallback без внешнего API.")
    return "\n".join(lines)
