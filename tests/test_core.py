from pathlib import Path

from src.analyzer import compare_documents
from src.models import Fragment, ParsedDocument
from src.parser import parse_pdf
from src.semantic import enhance_rows, validate_result
import src.semantic as semantic


SAMPLE = Path("data/sample")


def test_sample_pdf_extracts_text_and_clauses():
    path = next(SAMPLE.glob("*.pdf"))
    parsed = parse_pdf(path)
    assert parsed.fragments
    assert parsed.clauses
    assert any(fragment.clause_id == "3.4" for fragment in parsed.fragments)


def test_source_fragments_exist_for_sample_units():
    path = next(SAMPLE.glob("*.pdf"))
    parsed = parse_pdf(path)
    ids = {fragment.fragment_id for fragment in parsed.fragments}
    assert all(unit["fragment_id"] in ids for unit in parsed.units)


def make_doc(name, clauses):
    fragments = [Fragment(fragment_id=f"p1-c{k}", document_name=name, page_number=1, clause_id=k, raw_text=v, cleaned_text=v) for k, v in clauses.items()]
    return ParsedDocument(document_name=name, fragments=fragments, clauses={f.clause_id: f for f in fragments}, units=[], functions=[])


def test_new_removed_and_changed_clauses():
    before = make_doc("before", {"1.1": "old", "1.2": "same", "1.3": "removed"})
    after = make_doc("after", {"1.1": "new", "1.2": "same", "1.4": "added"})
    rows = {row.before.clause_id if row.before else row.after.clause_id: row for row in compare_documents(before, after)}
    assert rows["1.1"].status == "Изменена"
    assert rows["1.3"].status == "Удалена / не найдена в ПОСЛЕ"
    assert rows["1.4"].status == "Новая"


def test_fallback_does_not_require_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    before = make_doc("before", {"1.1": "same"})
    after = make_doc("after", {"1.1": "same"})
    assert compare_documents(before, after)[0].status == "Сохранена"


def test_semantic_fallback_does_not_turn_missing_clause_into_fact_loss(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    before = make_doc("before", {"2.4.1": "проводит проверку системы внутреннего контроля"})
    after = make_doc("after", {"2.4.9": "осуществляет проверку системы внутреннего контроля"})
    rows, candidates = enhance_rows(before, after, compare_documents(before, after), use_ai=False)
    assert candidates == 1
    assert rows[0].status in {"Изменена", "Потенциально перераспределена"}
    assert rows[0].semantic_status != "потенциально потеряна"


def test_ai_source_validation_rejects_unknown_fragment():
    try:
        validate_result({"before_fragment_id": "fake", "after_fragment_ids": ["p1-c2"]}, {"p1-c1"}, {"p1-c2"}, "p1-c1", "p1-c2")
    except ValueError:
        return
    raise AssertionError("invalid AI source was accepted")


def test_semantic_ai_path_uses_mock_and_validates_sources(monkeypatch):
    before = make_doc("before", {"2.4.1": "проверяет систему контроля"})
    after = make_doc("after", {"2.4.9": "осуществляет проверку системы контроля"})
    monkeypatch.setattr(semantic, "call_ai", lambda old, new: {
        "before_fragment_id": old.fragment_id,
        "after_fragment_ids": [new.fragment_id],
        "semantic_status": "смысл сохранен",
        "confidence": 0.93,
        "explanation": "Mock semantic match",
        "requires_human_review": False,
    })
    rows, _ = enhance_rows(before, after, compare_documents(before, after), use_ai=True)
    assert rows[0].analysis_method == "AI"
    assert rows[0].semantic_status == "смысл сохранен"


def test_auth_diagnostic_never_exposes_api_key(monkeypatch):
    fake_key = "sk-fake-secret-123456789"
    monkeypatch.setenv("OPENAI_API_KEY", fake_key)
    diagnostic = semantic._error_info(Exception(f"Incorrect API key provided: {fake_key}"))
    assert fake_key not in diagnostic["message"]
    assert "Authentication error: API key rejected by OpenAI." == diagnostic["message"]


def test_non_auth_diagnostic_redacts_credentials(monkeypatch):
    fake_key = "sk-fake-secret-987654321"
    monkeypatch.setenv("OPENAI_API_KEY", fake_key)
    diagnostic = semantic._error_info(Exception(f"request failed token={fake_key}"))
    assert fake_key not in diagnostic["message"]
    assert "[REDACTED]" in diagnostic["message"]


def test_semantic_status_and_result_type_consistency():
    base = {"before_fragment_id": "p1-c1", "after_fragment_ids": ["p1-c2"], "confidence": 0.95}
    saved = semantic.validate_result({**base, "semantic_status": "потенциально потеряна", "explanation": "no change in meaning"}, {"p1-c1"}, {"p1-c2"}, "p1-c1", "p1-c2")
    assert saved["semantic_status"] == "смысл сохранен"  # contradictory explanation is resolved safely
    assert saved["result_type"] == "FACT"
    assert saved["requires_human_review"] is False
    same = semantic.validate_result({**base, "semantic_status": "смысл сохранен", "explanation": "no change in meaning"}, {"p1-c1"}, {"p1-c2"}, "p1-c1", "p1-c2")
    assert same["result_type"] == "FACT"
    assert same["requires_human_review"] is False
    expanded = semantic.validate_result({**base, "semantic_status": "функция расширена", "explanation": "AFTER expands the scope"}, {"p1-c1"}, {"p1-c2"}, "p1-c1", "p1-c2")
    assert expanded["result_type"] == "FACT"
    uncertain = semantic.validate_result({**base, "semantic_status": "недостаточно данных / требуется проверка человеком", "explanation": "ambiguous"}, {"p1-c1"}, {"p1-c2"}, "p1-c1", "p1-c2")
    assert uncertain["result_type"] == "INFERENCE"
    assert uncertain["requires_human_review"] is True


def test_high_confidence_semantic_consistency_clears_false_review():
    normalized = semantic.validate_result({
        "before_fragment_id": "p1-c1", "after_fragment_ids": ["p1-c2"],
        "semantic_status": "недостаточно данных / требуется проверка человеком",
        "confidence": 0.95,
        "explanation": "The action is semantically consistent and the responsibility is maintained.",
        "requires_human_review": True,
    }, {"p1-c1"}, {"p1-c2"}, "p1-c1", "p1-c2")
    assert normalized["semantic_status"] == "смысл сохранен"
    assert normalized["result_type"] == "FACT"
    assert normalized["requires_human_review"] is False
