import re
from pathlib import Path

from pypdf import PdfReader

from .models import Fragment, ParsedDocument


CLAUSE_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+){1,3})(?:\.)?\s+")
SECTION_RE = re.compile(r"(?<![\w.])(\d+)(?:\.)?\s+")


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", " ").replace("\uf02d", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_pdf(path: str | Path) -> ParsedDocument:
    path = Path(path)
    reader = PdfReader(str(path))
    fragments: list[Fragment] = []
    clauses: dict[str, Fragment] = {}

    for page_number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        page_clauses = list(CLAUSE_RE.finditer(cleaned))
        if not page_clauses:
            fragment = Fragment(
                fragment_id=f"p{page_number}-body",
                document_name=path.name,
                page_number=page_number,
                raw_text=raw,
                cleaned_text=cleaned,
            )
            fragments.append(fragment)
            continue

        for index, match in enumerate(page_clauses):
            clause_id = match.group(1)
            start = match.start()
            end = page_clauses[index + 1].start() if index + 1 < len(page_clauses) else len(cleaned)
            fragment = Fragment(
                fragment_id=f"p{page_number}-c{clause_id}",
                document_name=path.name,
                page_number=page_number,
                clause_id=clause_id,
                raw_text=raw[start:] if index == len(page_clauses) - 1 else raw,
                cleaned_text=cleaned[start:end].strip(),
            )
            fragments.append(fragment)
            clauses.setdefault(clause_id, fragment)

    return ParsedDocument(
        document_name=path.name,
        fragments=fragments,
        clauses=clauses,
        units=extract_units(fragments),
        functions=extract_functions(fragments),
    )


def _fragment_text(fragments: list[Fragment], clause_id: str) -> Fragment | None:
    for fragment in fragments:
        if fragment.clause_id == clause_id:
            return fragment
    return None


def extract_units(fragments: list[Fragment]) -> list:
    result = []
    for fragment in fragments:
        if fragment.clause_id == "3.4":
            names = re.findall(r"(?:а|б|в|г)\.\s+(.+?)(?=\s+(?:а|б|в|г)\.\s+|$)", fragment.cleaned_text)
            for index, name in enumerate(names, start=1):
                result.append({"unit_name": name.strip(" ."), "fragment_id": fragment.fragment_id, "clause_id": "3.4", "index": index})
    return result


def extract_functions(fragments: list[Fragment]) -> list:
    result = []
    for fragment in fragments:
        clause = fragment.clause_id or ""
        if not (clause.startswith("2.4") or clause.startswith("4.") or clause.startswith("5.3") or clause.startswith("9.")):
            continue
        result.append({
            "function_id": f"{fragment.fragment_id}-function",
            "unit_name": None,
            "text": fragment.cleaned_text,
            "fragment_id": fragment.fragment_id,
            "clause_id": fragment.clause_id,
        })
    return result
