from pathlib import Path

from src.analyzer import compare_documents
from src.models import Fragment, ParsedDocument
from src.parser import parse_pdf


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
