"""Session-state helpers for keeping completed analysis data across reruns."""


def save_analysis_state(session_state, *, before, after, rows, structure, duplicates, table, recommendations, text):
    session_state.analysis_result = {"before": before, "after": after, "rows": rows, "structure": structure,
                                    "duplicates": duplicates, "table": table, "recommendations": recommendations, "text": text}
    session_state.analysis_completed = True
    session_state.uploaded_document_names = {"before": before.document_name, "after": after.document_name}


def get_analysis_state(session_state):
    return session_state.get("analysis_result")


def save_human_decision(session_state, fragment_id, decision, comment="", recommendation=""):
    decisions = dict(session_state.get("human_decisions", {}))
    decisions[fragment_id] = {"fragment_id": fragment_id, "ai_recommendation": recommendation,
                             "employee_decision": decision, "employee_comment": comment}
    session_state.human_decisions = decisions
    return decisions[fragment_id]
