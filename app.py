import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.analyzer import compare_documents, structural_changes
from src.parser import parse_pdf
from src.report import conclusion

load_dotenv()
st.set_page_config(page_title="AI-анализ документов", layout="wide")
st.title("AI-анализ организационной структуры и функционала")
st.caption("MVP: сравнение нормативных PDF с доказательствами из исходных фрагментов")

before_file = st.file_uploader("Документ ДО", type=["pdf"], key="before")
after_file = st.file_uploader("Документ ПОСЛЕ", type=["pdf"], key="after")


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
        progress.write("✓ Структура извлечена")
        progress.write("✓ Подразделения и функции определены")
        rows = compare_documents(before, after)
        progress.write("✓ Выполнено сопоставление и классификация")
        structure = structural_changes(before, after)
        progress.write("✓ Выполнена проверка рисков")
        fragments = {f.fragment_id: f for f in before.fragments + after.fragments}
        assert all((not r.before or r.before.fragment_id in fragments) and (not r.after or r.after.fragment_id in fragments) for r in rows)
        progress.write("✓ Источники проверены")
        ai_available = bool(os.getenv("OPENAI_API_KEY"))
        text = conclusion(before, after, rows, ai_available)
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
            "Статус": r.status, "Тип": r.evidence_type,
            "Объяснение": r.explanation, "Confidence": r.confidence,
        } for r in rows])
        st.dataframe(table, use_container_width=True, height=600)
        st.subheader("Потенциальные риски")
        risks = [r for r in rows if r.evidence_type == "RISK_FLAG"]
        if risks:
            for risk in risks:
                with st.expander(f"{risk.status}: {risk.before.clause_id if risk.before else risk.after.clause_id}"):
                    st.warning(risk.explanation)
                    st.write("ДО:", risk.before.text if risk.before else "нет")
                    st.write("ПОСЛЕ:", risk.after.text if risk.after else "нет")
        else:
            st.info("Автоматических risk flags по номерным пунктам не найдено.")
        st.subheader("Итоговое заключение")
        st.markdown(text)
        if not ai_available:
            st.info("OPENAI_API_KEY не задан: semantic analysis работает в deterministic fallback режиме.")
    except Exception as exc:
        progress.update(label="Анализ завершён с ошибкой", state="error")
        st.exception(exc)
