import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.analyzer import compare_documents, structural_changes
from src.parser import parse_pdf
from src.report import conclusion
from src.semantic import ai_status, check_ai_connection, duplicate_flags, enhance_rows, model_name
from src.models import FunctionRecord

load_dotenv()
st.set_page_config(page_title="AI-анализ документов", layout="wide")
st.title("AI-анализ организационной структуры и функционала")
st.caption("MVP: сравнение нормативных PDF с доказательствами из исходных фрагментов")

before_file = st.file_uploader("Документ ДО", type=["pdf"], key="before")
after_file = st.file_uploader("Документ ПОСЛЕ", type=["pdf"], key="after")
status = ai_status()
st.caption(f"AI status: {status['status']} | model: {status['model']}")
if status.get("error"):
    st.warning(f"Fallback reason: {status['error']['category']} ({status['error']['exception']}) — {status['error']['message']}")


def save_upload(uploaded):
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    handle.write(uploaded.getvalue())
    handle.close()
    return handle.name


if st.button("Провести анализ", type="primary", disabled=not (before_file and after_file)):
    progress = st.status("Выполняется agentic workflow", expanded=True)
    try:
        progress.write("✓ Документы загружены")
        before = parse_pdf(save_upload(before_file))
        after = parse_pdf(save_upload(after_file))
        def smoke_record(document, clause_id="5.3.1"):
            fragment = document.clauses.get(clause_id) or next(iter(document.clauses.values()))
            return FunctionRecord(function_id=fragment.fragment_id, text=fragment.cleaned_text, fragment_id=fragment.fragment_id, clause_id=fragment.clause_id)
        smoke = check_ai_connection(smoke_record(before), smoke_record(after))
        if smoke["status"] == "Connected":
            st.success(f"AI status: Connected | model: {smoke['model']} | smoke result: {smoke['result']['semantic_status']}")
        elif smoke.get("error"):
            st.warning(f"AI status: API error / fallback | model: {smoke['model']} | {smoke['error']['category']}: {smoke['error']['message']}")
        else:
            st.info(f"AI status: {smoke['status']} | model: {smoke['model']}")
        progress.write("✓ Извлечение документов")
        progress.write("✓ Разбор структуры")
        progress.write("✓ Извлечение функций")
        rows = compare_documents(before, after)
        progress.write("✓ Поиск кандидатов")
        rows, candidate_count = enhance_rows(before, after, rows, use_ai=True)
        progress.write(f"✓ Semantic AI matching (кандидатов: {candidate_count})")
        structure = structural_changes(before, after)
        progress.write("✓ Проверка потери функций")
        duplicates = duplicate_flags(after)
        progress.write("✓ Проверка дублирования")
        fragments = {f.fragment_id: f for f in before.fragments + after.fragments}
        assert all((not r.before or r.before.fragment_id in fragments) and (not r.after or r.after.fragment_id in fragments) for r in rows)
        progress.write("✓ Traceability validation")
        ai_available = bool(os.getenv("OPENAI_API_KEY"))
        text = conclusion(before, after, rows, ai_available, structure, duplicates)
        progress.write("✓ Заключение сформировано")
        progress.update(label="Анализ завершён", state="complete")

        st.subheader("Краткая сводка")
        st.write(text)
        st.subheader("Структурные изменения")
        st.dataframe(structure, use_container_width=True)
        st.subheader("Сравнительная таблица")
        import pandas as pd
        table = pd.DataFrame([{
            "Пункт ДО": r.before.clause_id if r.before else "",
            "Страница ДО": fragments[r.before.fragment_id].page_number if r.before else "",
            "Фрагмент ДО": r.before.text if r.before else "",
            "Пункт ПОСЛЕ": r.after.clause_id if r.after else "",
            "Страница ПОСЛЕ": fragments[r.after.fragment_id].page_number if r.after else "",
            "Фрагмент ПОСЛЕ": r.after.text if r.after else "",
            "Статус": r.status, "Semantic status": r.semantic_status or "",
            "Result type": r.result_type or r.evidence_type, "AI explanation": r.explanation,
            "Confidence": r.confidence, "Requires human review": r.requires_human_review,
            "Analysis method": r.analysis_method,
        } for r in rows])
        st.dataframe(table, use_container_width=True, height=600)
        st.subheader("Потенциальные риски")
        risks = [r for r in rows if r.evidence_type == "RISK_FLAG" and r.status != "Изменена"] + duplicates
        if risks:
            for risk in risks:
                with st.expander(f"{risk.status}: {risk.before.clause_id if risk.before else risk.after.clause_id}"):
                    st.warning(risk.explanation)
                    st.write("ДО:", risk.before.text if risk.before else "нет")
                    st.write("ПОСЛЕ:", risk.after.text if risk.after else "нет")
                    st.write("Метод:", risk.analysis_method, "Confidence:", risk.confidence)
        else:
            st.info("Автоматических risk flags по номерным пунктам не найдено.")
        st.subheader("Требуют проверки человеком")
        review_rows = [r for r in rows if r.requires_human_review]
        if review_rows:
            st.dataframe(pd.DataFrame([{
                "Semantic status": r.semantic_status or r.status,
                "Confidence": r.confidence,
                "Пункт ДО": r.before.clause_id if r.before else "",
                "Пункт ПОСЛЕ": r.after.clause_id if r.after else "",
                "Explanation": r.explanation,
            } for r in review_rows]), use_container_width=True)
        else:
            st.info("Результатов, требующих проверки человеком, не найдено.")
        st.subheader("Итоговое заключение")
        st.markdown(text)
        if not ai_available:
            st.info("OPENAI_API_KEY не задан: semantic analysis работает в deterministic fallback режиме.")
    except Exception as exc:
        progress.update(label="Анализ завершён с ошибкой", state="error")
        st.exception(exc)
