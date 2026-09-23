import difflib
import html
import os
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.analyzer import compare_documents, structural_changes
from src.decision_workflow import apply_employee_decision, decision_record, revalidate_records, russian_impact, russian_recommendation
from src.document_parser import parse_document
from src.models import FunctionRecord
from src.report import build_recommendations, conclusion, management_conclusion, review_key
from src.semantic import ai_status, check_ai_connection, duplicate_flags, enhance_rows, model_name
from src.ui_state import save_analysis_state, save_human_decision

load_dotenv()
st.set_page_config(page_title="AI-анализ документов", layout="wide")
st.title("AI-анализ организационной структуры и функционала")
st.caption("Сравнение редакций внутренних документов с доказательствами и экспертной проверкой.")

for key, default in (("expert_reviews", {}), ("decision_records", {}), ("human_decisions", {})):
    if key not in st.session_state:
        st.session_state[key] = default


def save_upload(uploaded):
    suffix = os.path.splitext(uploaded.name)[1].lower() or ".bin"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(uploaded.getvalue())
    handle.close()
    return handle.name


def restore_uploaded_name(parsed, original_name):
    parsed.document_name = original_name
    for fragment in parsed.fragments:
        fragment.document_name = original_name
    return parsed


def source_label(fragment, fragments):
    if not fragment:
        return "Источник не найден"
    stored = fragments.get(fragment.fragment_id)
    source = stored or fragment
    document_name = getattr(source, "document_name", None)
    if not document_name:
        return "Источник не найден"
    location = f"стр. {source.page_number}" if getattr(source, "page_number", None) else getattr(source, "source_locator", None)
    parts = [document_name]
    if location: parts.append(location)
    clause_id = getattr(source, "clause_id", None) or getattr(fragment, "clause_id", None)
    if clause_id: parts.append(f"пункт {clause_id}")
    return " · ".join(parts)


def diff_html(before_text, after_text):
    left_tokens, right_tokens = before_text.split(), after_text.split()
    left_out, right_out = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, left_tokens, right_tokens).get_opcodes():
        left, right = left_tokens[i1:i2], right_tokens[j1:j2]
        if tag == "equal":
            left_out.extend(html.escape(x) for x in left); right_out.extend(html.escape(x) for x in right)
        elif tag == "delete":
            left_out.extend(f"<span class='diff-removed'>{html.escape(x)}</span>" for x in left)
        elif tag == "insert":
            right_out.extend(f"<span class='diff-added'>{html.escape(x)}</span>" for x in right)
        else:
            left_out.extend(f"<span class='diff-removed'>{html.escape(x)}</span>" for x in left)
            right_out.extend(f"<span class='diff-added'>{html.escape(x)}</span>" for x in right)
    return " ".join(left_out), " ".join(right_out)


def russian_reason(row):
    return {"смысл сохранен":"Существенного изменения смысла функции не выявлено.", "функция уточнена":"Формулировка функции стала более точной или детализированной.", "функция расширена":"В новой редакции расширены действие или область ответственности функции.", "функция существенно изменена":"Содержание функции заметно отличается и требует внимания эксперта.", "потенциально перераспределена":"Функция могла быть перераспределена; требуется проверка.", "потенциально потеряна":"Аналог функции не подтверждён в новой редакции; требуется проверка.", "недостаточно данных / требуется проверка человеком":"Недостаточно данных для уверенного вывода; требуется экспертная проверка."}.get(row.semantic_status or row.status, "Изменение выявлено при сравнении документов.")


def run_analysis(before_file, after_file):
    progress = st.status("Выполняется анализ документов…", expanded=True)
    before = restore_uploaded_name(parse_document(save_upload(before_file)), before_file.name)
    after = restore_uploaded_name(parse_document(save_upload(after_file)), after_file.name)
    progress.write("✓ Чтение документов")
    progress.write("✓ Определение структуры и подразделений")
    rows = compare_documents(before, after)
    progress.write("✓ Сопоставление функций")
    rows, candidate_count = enhance_rows(before, after, rows, use_ai=True)
    progress.write("✓ Семантический AI-анализ изменений")
    structure = structural_changes(before, after)
    duplicates = duplicate_flags(after)
    progress.write("✓ Проверка источников и доказательств")
    fragments = {f.fragment_id: f for f in before.fragments + after.fragments}
    if not all((not r.before or r.before.fragment_id in fragments) and (not r.after or r.after.fragment_id in fragments) for r in rows):
        raise ValueError("Traceability validation failed")
    text = conclusion(before, after, rows, bool(os.getenv("OPENAI_API_KEY")), structure, duplicates)
    table = pd.DataFrame([{"Пункт ДО":r.before.clause_id if r.before else "", "Пункт ПОСЛЕ":r.after.clause_id if r.after else "", "Фрагмент ДО":r.before.text if r.before else "", "Фрагмент ПОСЛЕ":r.after.text if r.after else "", "Semantic status":r.semantic_status or "", "Result type":r.result_type or r.evidence_type, "Confidence":r.confidence, "Requires human review":r.requires_human_review, "AI explanation":r.explanation, "Analysis method":r.analysis_method} for r in rows])
    recommendations = build_recommendations(rows + duplicates)
    save_analysis_state(st.session_state, before=before, after=after, rows=rows, structure=structure, duplicates=duplicates, table=table, recommendations=recommendations, text=text)
    st.session_state.analysis_result["candidate_count"] = candidate_count
    progress.update(label="Анализ завершён", state="complete")


def render_analysis(state):
    before, after, rows, structure, duplicates = state["before"], state["after"], state["rows"], state["structure"], state["duplicates"]
    table, recommendations, text = state["table"], state["recommendations"], state["text"]
    fragments = {f.fragment_id:f for f in before.fragments + after.fragments}
    risks = [r for r in rows if r.result_type == "RISK_FLAG"] + duplicates
    reviews = [r for r in rows if r.requires_human_review]
    st.success("AI-анализ завершён")
    cols = st.columns(5)
    for col, label, value in zip(cols, ("Новых подразделений", "Сохранено подразделений", "Изменения для анализа", "Потенциальных рисков", "Требуют проверки"), (sum(x["status"]=="Новая" for x in structure), sum(x["status"]=="Сохранена" for x in structure), sum(bool(r.semantic_status and r.semantic_status != "смысл сохранен") for r in rows), len(risks), len(reviews))): col.metric(label, value)
    with st.expander("Техническая информация"):
        st.caption(f"Model: {model_name()} | AI status: {ai_status()['status']} | candidates: {state.get('candidate_count','—')}")
    tabs = st.tabs(["Обзор", "Структура", "Функции", "Риски и проверка", "Сравнение документов", "Заключение"])
    with tabs[0]:
        st.subheader("Результаты функционального анализа")
        st.dataframe(pd.DataFrame([{"Статус":k, "Количество":v} for k,v in table["Semantic status"].value_counts().items() if k]), hide_index=True, use_container_width=True)
        st.subheader("Изменение организационной структуры")
        a,b = st.columns(2)
        with a:
            st.markdown("**СТРУКТУРА ДО**")
            for x in structure:
                if x["status"] != "Новая": st.markdown(f"• {x['name']}")
        with b:
            st.markdown("**СТРУКТУРА ПОСЛЕ**")
            for x in structure:
                if x["status"] != "Удалена / не найдена в ПОСЛЕ": st.markdown(f"• {x['name']} — {x['status']}")
    with tabs[1]:
        for item in structure:
            with st.expander(f"{item['status']}: {item['name']}"):
                st.write(source_label(after.clauses.get("3.4") or before.clauses.get("3.4"), fragments))
    with tabs[2]:
        for _, item in table[table["Semantic status"].ne("")].head(20).iterrows():
            row = next((r for r in rows if (r.before and r.before.clause_id == item["Пункт ДО"]) or (r.after and r.after.clause_id == item["Пункт ПОСЛЕ"])), None)
            with st.expander(f"Пункт {item['Пункт ДО'] or item['Пункт ПОСЛЕ']} · {item['Semantic status']} · {item['Confidence']}"):
                st.write(russian_reason(row)); old,new = diff_html(item["Фрагмент ДО"], item["Фрагмент ПОСЛЕ"]); st.markdown("**ДО**"); st.markdown(old, unsafe_allow_html=True); st.markdown("**ПОСЛЕ**"); st.markdown(new, unsafe_allow_html=True)
    with tabs[3]:
        st.subheader("Потенциальные риски")
        if risks:
            for r in risks: st.warning(r.explanation)
        else: st.info("Подтверждённых автоматическим анализом потенциальных рисков не выявлено.")
        st.subheader("Требуют экспертной проверки")
        for review in reviews[:25]:
            source, key = review.before or review.after, review_key(review); current = st.session_state.expert_reviews.get(key,{})
            with st.expander(f"Пункт {source.clause_id if source else '—'} · Требует решения"):
                st.write(russian_reason(review)); st.write(russian_impact(review)); st.write("ДО:", review.before.text if review.before else "Не найдено"); st.write("ПОСЛЕ:", review.after.text if review.after else "Не найдено"); st.write("Рекомендация AI:", russian_recommendation(review)); st.write("Источник:", source_label(source, fragments))
                options=["Не принято","Принять рекомендацию AI","Отклонить рекомендацию","Изменить решение","Отложить на дополнительную проверку"]
                decision=st.selectbox("Решение сотрудника", options, index=options.index(current.get("decision","Не принято")) if current.get("decision","Не принято") in options else 0, key=f"decision_{key}")
                comment=st.text_area("Комментарий сотрудника", value=current.get("comment",""), key=f"comment_{key}")
                if st.button("Сохранить решение", key=f"save_{key}"):
                    base=st.session_state.decision_records.get(key) or decision_record(review,russian_recommendation(review)); st.session_state.decision_records[key]=apply_employee_decision(base,decision,comment); st.session_state.expert_reviews[key]={"ai_result":review.model_dump(),"decision":decision,"comment":comment}; save_human_decision(st.session_state,key,decision,comment,russian_recommendation(review)); st.success("Решение сохранено"); st.rerun()
    with tabs[4]: st.dataframe(table, use_container_width=True, height=600)
    with tabs[5]:
        st.markdown(management_conclusion(rows + duplicates, structure, recommendations, st.session_state.human_decisions))


u1,u2=st.columns(2)
with u1: before_file=st.file_uploader("ДО · текущая редакция", type=["pdf","docx","xlsx"], key="before")
with u2: after_file=st.file_uploader("ПОСЛЕ · новая редакция", type=["pdf","docx","xlsx"], key="after")
if st.button("Запустить AI-анализ", type="primary", disabled=not(before_file and after_file), use_container_width=True):
    run_analysis(before_file, after_file)
if st.session_state.get("analysis_result") is not None:
    render_analysis(st.session_state["analysis_result"])
