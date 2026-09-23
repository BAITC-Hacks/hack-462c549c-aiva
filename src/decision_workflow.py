"""Pure decision-loop helpers. AI findings remain immutable; decisions are separate."""


def decision_record(row, recommendation: str) -> dict:
    source = row.before or row.after
    return {
        "ai_finding": row.explanation,
        "ai_classification": row.semantic_status or row.status,
        "ai_recommendation": recommendation,
        "employee_decision": "Не проверено",
        "employee_override": None,
        "employee_comment": "",
        "effective_decision": None,
        "source_fragment_ids": [x.fragment_id for x in (row.before, row.after) if x],
        "ai_result": row.model_dump(),
    }


def apply_employee_decision(record: dict, decision: str, comment: str = "", override: str | None = None) -> dict:
    updated = dict(record)
    updated["employee_decision"] = decision
    updated["employee_override"] = override if decision == "Изменить решение" else None
    updated["employee_comment"] = comment
    updated["effective_decision"] = override if decision == "Изменить решение" and override else decision
    return updated


def revalidate_records(records: list[dict]) -> dict:
    accepted = [r for r in records if r.get("effective_decision") in {"Принять рекомендацию AI", "Изменено", "Подтверждено"}]
    rejected = [r for r in records if r.get("effective_decision") in {"Отклонить рекомендацию", "Отклонено"}]
    deferred = [r for r in records if r.get("effective_decision") in {None, "Не проверено", "Отложить на дополнительную проверку"}]
    resolved = [r for r in accepted if r.get("ai_classification") in {"потенциально потеряна", "потенциально перераспределена", "потенциальное дублирование"}]
    return {
        "accepted": len(accepted), "rejected": len(rejected), "changed": sum(r.get("employee_decision") == "Изменить решение" for r in records),
        "resolved": len(resolved), "residual": len(deferred), "new_conflicts": 0,
        "needs_review": len(deferred), "resolved_records": resolved, "residual_records": deferred,
    }


def russian_recommendation(row) -> str:
    status = row.semantic_status or row.status
    return {
        "потенциально потеряна": "Проверить возможность восстановления или закрепления функции.",
        "потенциально перераспределена": "Проверить корректность закрепления ответственности.",
        "потенциальное дублирование": "Проверить возможное пересечение функций.",
        "функция существенно изменена": "Провести дополнительную экспертную проверку изменения.",
    }.get(status, "Требуется экспертное решение.")


def russian_impact(row) -> str:
    status = row.semantic_status or row.status
    if status in {"потенциально потеряна", "потенциально перераспределена"}:
        return "Изменение может повлиять на закрепление функции и требует проверки сотрудником."
    if status in {"потенциальное дублирование", "потенциальный конфликт полномочий"}:
        return "Изменение может привести к пересечению ответственности; требуется проверка."
    if row.requires_human_review:
        return "Данных недостаточно для однозначного вывода; требуется экспертное решение."
    return "Существенного риска по имеющимся источникам не установлено."
