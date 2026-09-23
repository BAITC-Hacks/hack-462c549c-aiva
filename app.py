import os
import tempfile
import html
import difflib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.analyzer import compare_documents, structural_changes
from src.document_parser import parse_document
from src.report import build_recommendations, conclusion, expert_conclusion, review_key, review_priority
from src.semantic import ai_status, check_ai_connection, duplicate_flags, enhance_rows, model_name
from src.models import FunctionRecord
from src.decision_workflow import apply_employee_decision, decision_record, revalidate_records, russian_impact, russian_recommendation

load_dotenv()
st.set_page_config(page_title="AI-анализ документов", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1500px; padding-top: 2rem; padding-bottom: 3rem;}
.hero-subtitle {color:#667085; font-size:1.05rem; margin-bottom:.35rem;}
.brand-line {color:#175cd3; font-size:.9rem; margin-bottom:1rem;}
.badge {display:inline-block; padding:.28rem .65rem; border:1px solid #bfdbfe; border-radius:999px; margin-right:.35rem; color:#175cd3; background:#eff8ff; font-size:.8rem;}
div[data-testid="stMetric"] {border:1px solid #eaecf0; border-radius:14px; padding:1rem; background:#fff; box-shadow:0 1px 2px rgba(16,24,40,.04);}
.section-note {color:#667085; font-size:.9rem;}
.diff-added {background:#dcfce7; color:#166534; padding:1px 3px; border-radius:3px;}
.diff-removed {background:#fee2e2; color:#991b1b; text-decoration:line-through; padding:1px 3px; border-radius:3px;}
</style>
""", unsafe_allow_html=True)
st.title("AI-анализ организационной структуры и функционала")
st.markdown('<div class="hero-subtitle">Сравнение редакций внутренних документов с выявлением структурных и функциональных изменений, доказательствами и экспертной проверкой.</div><span class="badge">AI-assisted</span><span class="badge">Evidence-based</span><span class="badge">Human-in-the-loop</span>', unsafe_allow_html=True)
st.markdown('<div class="brand-line">Решение для АО «Казахтелеком»</div>', unsafe_allow_html=True)
if "expert_reviews" not in st.session_state:
    st.session_state.expert_reviews = {}
if "decision_records" not in st.session_state:
    st.session_state.decision_records = {}

upload_cols = st.columns(2)
with upload_cols[0]:
    before_file = st.file_uploader("ДО · текущая редакция", type=["pdf", "docx", "xlsx"], key="before")
    if before_file: st.success("Документ загружен")
with upload_cols[1]:
    after_file = st.file_uploader("ПОСЛЕ · новая редакция", type=["pdf", "docx", "xlsx"], key="after")
    if after_file: st.success("Документ загружен")
status = ai_status()
with st.expander("Техническая информация"):
    st.caption(f"AI status: {status['status']} | model: {status['model']}")
    if status.get("error"):
        st.warning(f"Fallback reason: {status['error']['category']} ({status['error']['exception']}) — {status['error']['message']}")


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
    location = f"стр. {stored.page_number}" if stored and stored.page_number else (stored.source_locator if stored else "местоположение не определено")
    return f"{fragment.document_name} · {location} · пункт {fragment.clause_id or '—'}"


def russian_reason(row):
    status = row.semantic_status or row.status
    return {
        "смысл сохранен": "Существенного изменения смысла функции не выявлено.",
        "редакционное изменение": "Изменена формулировка без подтверждённого изменения содержания.",
        "функция уточнена": "Формулировка функции стала более точной или детализированной.",
        "функция расширена": "В новой редакции расширены действие или область ответственности функции.",
        "функция сокращена": "В новой редакции часть содержания функции могла быть сокращена.",
        "функция существенно изменена": "Содержание функции заметно отличается и требует внимания эксперта.",
        "потенциально перераспределена": "Функция могла быть перераспределена; требуется проверка.",
        "потенциально потеряна": "Аналог функции не подтверждён в новой редакции; требуется проверка.",
        "недостаточно данных / требуется проверка человеком": "Недостаточно данных для уверенного вывода; требуется экспертная проверка.",
    }.get(status, "Изменение выявлено при сравнении документов.")


def diff_html(before_text, after_text):
    left_tokens, right_tokens = before_text.split(), after_text.split()
    left_out, right_out = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, left_tokens, right_tokens).get_opcodes():
        left, right = left_tokens[i1:i2], right_tokens[j1:j2]
        if tag == "equal":
            left_out.extend(html.escape(x) for x in left); right_out.extend(html.escape(x) for x in right)
        elif tag == "delete":
            left_out.extend(f'<span class="diff-removed">{html.escape(x)}</span>' for x in left)
        elif tag == "insert":
            right_out.extend(f'<span class="diff-added">{html.escape(x)}</span>' for x in right)
        else:
            left_out.extend(f'<span class="diff-removed">{html.escape(x)}</span>' for x in left)
            right_out.extend(f'<span class="diff-added">{html.escape(x)}</span>' for x in right)
    return " ".join(left_out), " ".join(right_out)


if st.button("Запустить AI-анализ", type="primary", disabled=not (before_file and after_file), use_container_width=True):
    progress = st.status("Выполняется agentic workflow", expanded=True)
    try:
        progress.write("✓ Документы загружены")
        before = restore_uploaded_name(parse_document(save_upload(before_file)), before_file.name)
        after = restore_uploaded_name(parse_document(save_upload(after_file)), after_file.name)
        def smoke_record(document, clause_id="5.3.1"):
            fragment = document.clauses.get(clause_id) or next(iter(document.clauses.values()))
            return FunctionRecord(function_id=fragment.fragment_id, text=fragment.cleaned_text, fragment_id=fragment.fragment_id, clause_id=fragment.clause_id)
        smoke = check_ai_connection(smoke_record(before), smoke_record(after))
        if smoke["status"] == "Connected":
            st.success("AI-анализ подключён")
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
        recommendations = build_recommendations(rows + duplicates)
        display_labels = {"смысл сохранен":"Смысл сохранён", "редакционное изменение":"Редакционное изменение", "функция уточнена":"Функция уточнена", "функция расширена":"Функция расширена", "функция сокращена":"Функция сокращена", "функция существенно изменена":"Существенно изменена", "потенциально перераспределена":"Потенциально перераспределена", "потенциально потеряна":"Потенциально потеряна", "недостаточно данных / требуется проверка человеком":"Требует проверки"}
        semantic_counts = {display_labels[k]: v for k, v in table["Semantic status"].value_counts().to_dict().items() if k in display_labels} if not table.empty else {}
        risks = [r for r in rows if r.result_type == "RISK_FLAG"] + duplicates
        review_rows = [r for r in rows if r.requires_human_review]
        prioritized = [(r, review_priority(r)) for r in rows if r.requires_human_review or r.result_type == "RISK_FLAG"]
        new_units = sum(x["status"] == "Новая" for x in structure)
        saved_units = sum(x["status"] == "Сохранена" for x in structure)
        changed_functions = sum(bool(r.semantic_status and r.semantic_status not in {"смысл сохранен"}) for r in rows)

        st.success("AI-анализ завершён")
        st.subheader("Ключевые результаты")
        metrics = st.columns(5)
        metrics[0].metric("Новых подразделений", new_units)
        metrics[1].metric("Сохранено подразделений", saved_units)
        metrics[2].metric("Изменения для анализа", changed_functions)
        metrics[3].metric("Потенциальных рисков", len(risks))
        metrics[4].metric("Требуют проверки", len(review_rows))
        with st.expander("Техническая информация"):
            st.caption(f"Model: {model_name()} | AI status: {ai_status()['status']} | candidates: {candidate_count}")

        tabs = st.tabs(["Обзор", "Структура", "Функции", "Риски и проверка", "Сравнение документов", "Заключение"])
        with tabs[0]:
            st.markdown("Документ → структурный анализ → семантическое сравнение → проверка доказательств → экспертная валидация")
            st.subheader("Результаты функционального анализа")
            st.dataframe(pd.DataFrame([{"Статус": k, "Количество": v} for k, v in semantic_counts.items()]), hide_index=True, use_container_width=True)
            st.subheader("Приоритет экспертной проверки")
            priority_counts = {}
            for _, p in prioritized: priority_counts[p["priority"]] = priority_counts.get(p["priority"], 0) + 1
            if priority_counts:
                for label, count in priority_counts.items():
                    st.markdown(f"**{label.upper()}**  \n{count} случаев")
                    st.caption("Неоднозначные изменения направлены на экспертную валидацию.")
            else:
                st.info("Неоднозначных результатов нет.")
            st.subheader("Изменение организационной структуры")
            st.info("Перераспределение ответственности автоматически не подтверждено. Для вывода о переносе конкретных функций требуется подтверждённое сопоставление функций подразделений.")
            map_cols = st.columns(2)
            with map_cols[0]:
                st.markdown("**СТРУКТУРА ДО**")
                for item in structure:
                    if item["status"] != "Новая": st.markdown(f"• {item['name']}")
            with map_cols[1]:
                st.markdown("**СТРУКТУРА ПОСЛЕ**")
                for item in structure:
                    if item["status"] == "Новая": st.markdown(f"• {item['name']} — **НОВОЕ**")
                    elif item["status"] == "Сохранена": st.markdown(f"• {item['name']} — **СОХРАНЕНО**")
            st.caption("Перенос функций между подразделениями не визуализируется без достаточного подтверждения источниками.")
        with tabs[1]:
            st.subheader("Структурные изменения")
            for item in structure:
                fragment = after.clauses.get("3.4") or before.clauses.get("3.4")
                with st.expander(f"{item['status']}: {item['name']}"):
                    with st.expander("Показать источник"):
                        st.write(source_label(fragment, fragments))
        with tabs[2]:
            st.subheader("Функциональный анализ")
            meaningful = table[table["Semantic status"].isin(["функция расширена", "функция уточнена", "функция сокращена", "функция существенно изменена", "потенциально перераспределена", "потенциально потеряна"])]
            for _, item in meaningful.head(20).iterrows():
                with st.expander(f"Пункт {item['Пункт ДО'] or item['Пункт ПОСЛЕ']} · {display_labels.get(item['Semantic status'], item['Semantic status']).upper()} · Уверенность: {item['Confidence']}"):
                    row = next((r for r in rows if (r.before and r.before.clause_id == item['Пункт ДО']) or (r.after and r.after.clause_id == item['Пункт ПОСЛЕ'])), None)
                    st.markdown("**Что изменилось**")
                    st.write(russian_reason(row) if row else "Изменение выявлено при сравнении документов.")
                    old_diff, new_diff = diff_html(item["Фрагмент ДО"][:600], item["Фрагмент ПОСЛЕ"][:600])
                    st.markdown("**ДО**")
                    st.markdown(old_diff, unsafe_allow_html=True)
                    st.markdown("**ПОСЛЕ**")
                    st.markdown(new_diff, unsafe_allow_html=True)
                    with st.expander("Почему AI сделал такой вывод"):
                        st.write(item["AI explanation"])
            with st.expander("Показать все результаты анализа"):
                st.dataframe(table, use_container_width=True, height=500)
        with tabs[3]:
            st.subheader("Потенциальные риски")
            if risks:
                for risk in risks:
                    source = risk.before or risk.after
                    with st.expander(f"{risk.status}: {source.clause_id if source else '—'}"):
                        st.warning(risk.explanation)
                        st.write("Основание:", source_label(source, fragments))
            else:
                st.info("Подтверждённых автоматическим анализом потенциальных рисков не выявлено.")
            st.subheader("Требуют экспертной проверки")
            st.caption("AI-агент направляет неоднозначные случаи на экспертную проверку вместо неподтверждённого вывода.")
            for review in review_rows[:25]:
                source = review.before or review.after
                key = review_key(review)
                priority = review_priority(review)
                with st.expander(f"Пункт {source.clause_id if source else '—'} · {priority['priority'].upper()} · Уверенность: {review.confidence}"):
                    st.markdown("**Причина**")
                    st.write(priority["reason"])
                    st.markdown("**Что обнаружено**")
                    st.write(russian_reason(review))
                    st.markdown("**Возможное последствие**")
                    st.write(russian_impact(review))
                    st.markdown("**Редакция ДО**")
                    st.write(review.before.text if review.before else "Не найдено")
                    st.markdown("**Редакция ПОСЛЕ**")
                    st.write(review.after.text if review.after else "Не найдено")
                    if review.before and review.after:
                        old_diff, new_diff = diff_html(review.before.text, review.after.text)
                        st.markdown("**Визуальное сравнение**")
                        st.markdown("ДО: " + old_diff, unsafe_allow_html=True)
                        st.markdown("ПОСЛЕ: " + new_diff, unsafe_allow_html=True)
                    st.markdown("**Рекомендация AI**")
                    st.write(russian_recommendation(review))
                    st.markdown(f"**Основание:** стр. {fragments[source.fragment_id].page_number if source and source.fragment_id in fragments else '—'} · пункт {source.clause_id if source else '—'}")
                    current = st.session_state.expert_reviews.get(key, {})
                    options = ["Не проверено", "Принять рекомендацию AI", "Отклонить рекомендацию", "Изменить решение", "Отложить на дополнительную проверку"]
                    decision = st.selectbox("Решение сотрудника", options, index=options.index(current.get("decision", "Не проверено")) if current.get("decision", "Не проверено") in options else 0, key=f"decision_{key}")
                    comment = st.text_area("Комментарий эксперта", value=current.get("comment", ""), key=f"comment_{key}")
                    override = st.text_input("Решение сотрудника", value=current.get("override", ""), key=f"override_{key}") if decision == "Изменить решение" else None
                    if st.button("Сохранить решение", key=f"save_{key}"):
                        base = st.session_state.decision_records.get(key) or decision_record(review, russian_recommendation(review))
                        st.session_state.decision_records[key] = apply_employee_decision(base, decision, comment, override)
                        st.session_state.expert_reviews[key] = {"ai_result": review.model_dump(), "decision": decision, "comment": comment, "override": override}
                        st.success("Решение сохранено. AI-результат не изменён.")
            if len(review_rows) > 25:
                st.caption(f"Показаны первые 25 из {len(review_rows)} случаев.")
        with tabs[4]:
            st.subheader("Сравнение документов")
            status_filter = st.multiselect("Семантический статус", sorted(x for x in table["Semantic status"].dropna().unique() if x), format_func=lambda x: display_labels.get(x, x))
            type_filter = st.multiselect("Тип вывода", ["FACT", "INFERENCE", "RISK_FLAG"])
            only_review = st.checkbox("Только экспертная проверка")
            filtered = table
            if status_filter: filtered = filtered[filtered["Semantic status"].isin(status_filter)]
            if type_filter: filtered = filtered[filtered["Result type"].isin(type_filter)]
            if only_review: filtered = filtered[filtered["Requires human review"]]
            st.dataframe(filtered, use_container_width=True, height=600)
        with tabs[5]:
            st.subheader("Итоговое заключение")
            st.markdown(text)
            st.subheader("Рекомендации AI-агента")
            if recommendations:
                for rec in recommendations[:25]:
                    review = st.session_state.expert_reviews.get(rec["source"], {})
                    review_status = {"Подтверждено": "Подтверждено экспертом", "Отклонено": "Отклонено экспертом", "Не проверено": "Ожидает экспертной проверки"}.get(review.get("decision", "Не проверено"), "Ожидает экспертной проверки")
                    st.write(f"**{rec['priority']}** — {rec['action']} — {review_status}")
                    st.caption(f"Наблюдение: {rec['observation']} | source: {rec['source']} | confidence: {rec['confidence']}")
            else:
                st.info("Дополнительных действий по результатам автоматического анализа не сформировано.")
            st.subheader("Заключение после экспертной проверки")
            for item in expert_conclusion(rows + duplicates, st.session_state.expert_reviews)[:25]:
                st.write(f"**{item['status']}** — {item['text']} ({item['source']})")
            records = list(st.session_state.decision_records.values())
            if any(r.get("effective_decision") in {"Принять рекомендацию AI", "Изменено"} for r in records):
                if st.button("Применить решения и провести повторную проверку"):
                    st.session_state.revalidation = revalidate_records(records)
            if st.session_state.get("revalidation"):
                result = st.session_state.revalidation
                st.subheader("Результат повторной проверки")
                st.write(f"Принято решений: {result['accepted']} · Отклонено: {result['rejected']} · Изменено: {result['changed']} · Устранено потенциальных рисков: {result['resolved']} · Осталось потенциальных рисков: {result['residual']} · Новых конфликтов: {result['new_conflicts']} · Требуют проверки: {result['needs_review']}")
        if not ai_available:
            st.info("OPENAI_API_KEY не задан: semantic analysis работает в deterministic fallback режиме.")
    except Exception as exc:
        progress.update(label="Анализ завершён с ошибкой", state="error")
        st.exception(exc)
