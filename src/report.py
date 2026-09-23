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


def _decision_value(state: dict | None) -> str | None:
    """Read the employee decision without exposing the internal state shape."""
    if not state:
        return None
    return state.get("employee_decision") or state.get("decision") or state.get("effective_decision")


def _clause_sort_key(clause_id: str):
    parts = clause_id.rstrip(".").split(".")
    if all(part.isdigit() for part in parts):
        return (0, tuple(int(part) for part in parts))
    return (1, clause_id.lower())


def format_clause_ids(clause_ids: list[str]) -> str:
    """Format clause IDs, compressing only truly contiguous siblings."""
    unique = sorted({str(value).strip() for value in clause_ids if value}, key=_clause_sort_key)
    if not unique:
        return "не указаны"

    groups = []
    current = [unique[0]]
    for clause_id in unique[1:]:
        previous = current[-1]
        previous_parts = previous.rstrip(".").split(".")
        current_parts = clause_id.rstrip(".").split(".")
        contiguous = (
            len(previous_parts) == len(current_parts)
            and previous_parts[:-1] == current_parts[:-1]
            and previous_parts[-1].isdigit()
            and current_parts[-1].isdigit()
            and int(current_parts[-1]) == int(previous_parts[-1]) + 1
        )
        if contiguous:
            current.append(clause_id)
        else:
            groups.append(current)
            current = [clause_id]
    groups.append(current)

    formatted = []
    for group in groups:
        if len(group) >= 2:
            formatted.append(f"{group[0]}–{group[-1]}")
        else:
            formatted.append(group[0])
    return ", ".join(formatted)


def _recommendation_group_key(action: str) -> str:
    normalized = " ".join((action or "").lower().split()).strip(" .")
    if "ответствен" in normalized and ("закреп" in normalized or "подраздел" in normalized):
        return "responsibility_assignment"
    return normalized


def group_recommendations(recommendations: list[dict], rows: list[ComparisonRow], human_decisions=None) -> list[dict]:
    """Group repeated recommendations for the management-facing conclusion.

    Fragment IDs remain available in the returned internal data, while the
    rendered conclusion uses only business clause IDs.
    """
    human_decisions = human_decisions or {}
    row_by_source = {}
    for row in rows:
        for fragment in (row.before, row.after):
            if fragment:
                row_by_source.setdefault(fragment.fragment_id, []).append(row)

    grouped = {}
    for recommendation in recommendations:
        key = _recommendation_group_key(recommendation.get("action", ""))
        entry = grouped.setdefault(key, {
            "group_key": key,
            "action": recommendation.get("action", ""),
            "reason": "Выявлены результаты, для которых требуется однотипная экспертная проверка.",
            "source_fragment_ids": [],
            "rows": [],
            "recommendations": [],
        })
        if key == "responsibility_assignment":
            entry["action"] = "Проверить корректность закрепления ответственности между подразделениями."
            entry["reason"] = "Выявлены изменения организационной структуры и распределения ответственных ролей."
        entry["recommendations"].append(recommendation)
        source_id = recommendation.get("source")
        if source_id and source_id not in entry["source_fragment_ids"]:
            entry["source_fragment_ids"].append(source_id)
        for row in row_by_source.get(source_id, []):
            if row not in entry["rows"]:
                entry["rows"].append(row)

    result = []
    for entry in grouped.values():
        clause_ids = []
        decisions = []
        for row in entry["rows"]:
            source = row.before or row.after
            if source and source.clause_id:
                clause_ids.append(source.clause_id)
            decision = _decision_value(human_decisions.get(review_key(row)))
            if decision:
                decisions.append(decision)
        if decisions and all(value in {"Принять рекомендацию AI", "Подтверждено"} for value in decisions):
            status = "Принято сотрудником"
        elif decisions and all(value in {"Отклонить рекомендацию", "Отклонить рекомендацию AI", "Отклонено"} for value in decisions):
            status = "Отклонено сотрудником"
        elif decisions and all(value in {"Требуется уточнение", "Отложить на дополнительную проверку"} for value in decisions):
            status = "Требует уточнения"
        elif decisions:
            status = "Частично проверено"
        else:
            status = "Ожидает экспертной проверки"
        result.append({
            "group_key": entry["group_key"],
            "action": entry["action"],
            "reason": entry["reason"],
            "clause_ids": sorted(set(clause_ids), key=_clause_sort_key),
            "affected_clauses": format_clause_ids(clause_ids),
            "status": status,
            "source_fragment_ids": entry["source_fragment_ids"],
        })
    return result


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


def management_conclusion(rows, structure, recommendations, human_decisions):
    required = []
    seen = set()
    for row in rows:
        if (row.requires_human_review or row.result_type == "RISK_FLAG") and review_key(row) not in seen:
            required.append(row)
            seen.add(review_key(row))
    checked = sum(_decision_value(human_decisions.get(review_key(row))) not in {None, "", "Не принято", "Не проверено"} for row in required)
    accepted = sum(_decision_value(human_decisions.get(review_key(row))) in {"Принять рекомендацию AI", "Подтверждено"} for row in required)
    rejected = sum(_decision_value(human_decisions.get(review_key(row))) in {"Отклонить рекомендацию", "Отклонить рекомендацию AI", "Отклонено"} for row in required)
    deferred = sum(_decision_value(human_decisions.get(review_key(row))) in {"Требуется уточнение", "Отложить на дополнительную проверку"} for row in required)
    preliminary = bool(required) and checked < len(required)
    if preliminary:
        title = "ПРЕДВАРИТЕЛЬНОЕ"
    elif required:
        title = "ЗАКЛЮЧЕНИЕ С УЧЁТОМ ЭКСПЕРТНОЙ ПРОВЕРКИ"
    else:
        title = "ИТОГОВОЕ ЗАКЛЮЧЕНИЕ"
    lines = [title, "", "1. Результат анализа", "Система сопоставила структурные пункты и функциональные формулировки двух редакций документов.", "", "2. Ключевые структурные изменения"]
    lines.extend(f"- {item['status']}: {item['name']}" for item in structure[:8])
    lines += ["", "3. Ключевые функциональные изменения"]
    counts = {}
    for row in rows:
        if row.semantic_status: counts[row.semantic_status] = counts.get(row.semantic_status, 0) + 1
    for key in ("смысл сохранен", "функция уточнена", "функция расширена", "функция сокращена", "функция существенно изменена", "потенциально перераспределена", "потенциально потеряна"):
        if counts.get(key): lines.append(f"- {key}: {counts[key]}")
    lines += ["", "4. Вопросы, требующие экспертного решения", f"Направлено на проверку: {len(required)}; проверено: {checked}; принято: {accepted}; отклонено: {rejected}; требует уточнения: {deferred}; ожидает решения: {len(required)-checked}", "", "5. Потенциальные риски"]
    risk_count = sum(r.result_type == "RISK_FLAG" for r in rows)
    lines.append(f"Выявлено {risk_count} потенциальных риск-флагов; они требуют экспертного подтверждения." if risk_count else f"Автоматически подтверждённых случаев потенциальной потери, дублирования или конфликта функций не выявлено. При этом {len(required)} результатов требуют экспертной проверки.")
    lines += ["", "6. Рекомендации"]
    grouped = group_recommendations(recommendations, rows, human_decisions)
    if grouped:
        for recommendation in grouped:
            lines.extend([
                f"- {recommendation['action']}",
                f"  Основание: {recommendation['reason']}",
                f"  Затронутые пункты: {recommendation['affected_clauses']}",
                f"  Статус: {recommendation['status']}",
            ])
    else:
        lines.append("Дополнительных действий по результатам автоматического анализа не сформировано.")
    lines += ["", "7. Статус экспертной проверки"]
    if not required:
        lines.append("Экспертная проверка не требуется для автоматически подтверждённых результатов.")
    elif checked == 0:
        lines.append("Экспертная проверка ещё не начата.")
        lines.append("Окончательное заключение формируется после завершения экспертной проверки.")
    elif preliminary:
        lines.append(f"Экспертная проверка не завершена. Проверено: {checked} из {len(required)}.")
        lines.append("Окончательное заключение будет сформировано после рассмотрения всех результатов, требующих экспертного решения.")
    else:
        lines.append("Экспертная проверка завершена для всех направленных результатов.")
        lines += ["", "8. Решения сотрудника", f"- Принято рекомендаций: {accepted}", f"- Отклонено рекомендаций: {rejected}", f"- Требует уточнения: {deferred}"]
        accepted_rows = [row for row in required if _decision_value(human_decisions.get(review_key(row))) in {"Принять рекомендацию AI", "Подтверждено"}]
        rejected_rows = [row for row in required if _decision_value(human_decisions.get(review_key(row))) in {"Отклонить рекомендацию", "Отклонить рекомендацию AI", "Отклонено"}]
        if accepted_rows:
            lines.append(f"Подтверждённые выводы по пунктам: {format_clause_ids([((row.before or row.after).clause_id) for row in accepted_rows if (row.before or row.after) and (row.before or row.after).clause_id])}.")
        if rejected_rows:
            lines.append(f"Отклонённые выводы по пунктам: {format_clause_ids([((row.before or row.after).clause_id) for row in rejected_rows if (row.before or row.after) and (row.before or row.after).clause_id])}.")
        confirmed_risks = [row for row in required if row.result_type == "RISK_FLAG" and _decision_value(human_decisions.get(review_key(row))) in {"Принять рекомендацию AI", "Подтверждено"}]
        if confirmed_risks:
            lines.append(f"Подтверждённые потенциальные риски по пунктам: {format_clause_ids([((row.before or row.after).clause_id) for row in confirmed_risks if (row.before or row.after) and (row.before or row.after).clause_id])}.")
    return "\n".join(lines)
