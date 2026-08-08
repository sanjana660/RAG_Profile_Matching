from __future__ import annotations

import json
import os
import re
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SECTION_PATTERN = re.compile(
    r"^\s*(summary|profile|skills|experience|work experience|professional experience|education|projects|certifications)\s*:??\s*$",
    re.IGNORECASE,
)

SKILL_HINTS = [
    "python",
    "sql",
    "machine learning",
    "nlp",
    "llm",
    "rag",
    "data analysis",
    "data engineering",
    "spark",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "pandas",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "javascript",
    "react",
    "java",
    "c#",
    "go",
    "fastapi",
    "django",
    "flask",
    "tableau",
    "power bi",
]


@dataclass
class ResumeChunk:
    candidate_name: str
    resume_path: str
    section: str
    chunk_text: str
    chunk_index: int
    skills: list[str]
    experience_years: float | None
    education: list[str]
    source_text: str


class EmbeddingBackend:
    def __init__(self, provider: str = "hf", model_name: str | None = None):
        self.provider = provider.lower()
        self.model_name = model_name or self._default_model_name()
        self._model = None
        self._vectorizer = None
        self._backend = "tfidf"

    def _default_model_name(self) -> str:
        if self.provider == "openai":
            return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        if self.provider == "cohere":
            return os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
        return os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    def fit(self, texts: list[str]) -> None:
        if self.provider == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4096)
            self._vectorizer.fit(texts)
            self._backend = "tfidf"
            return

        if self.provider == "hf":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                self._backend = "sentence_transformers"
                return
            except Exception:
                self._backend = "tfidf"
        elif self.provider == "openai":
            try:
                from openai import OpenAI

                self._model = OpenAI()
                self._backend = "openai"
                return
            except Exception:
                self._backend = "tfidf"
        elif self.provider == "cohere":
            try:
                import cohere

                self._model = cohere.Client(os.getenv("COHERE_API_KEY", ""))
                self._backend = "cohere"
                return
            except Exception:
                self._backend = "tfidf"

        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4096)
        self._vectorizer.fit(texts)
        self._backend = "tfidf"

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._backend == "sentence_transformers":
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return np.asarray(vectors, dtype=np.float32)

        if self._backend == "openai":
            response = self._model.embeddings.create(model=self.model_name, input=texts)
            vectors = [item.embedding for item in response.data]
            return np.asarray(vectors, dtype=np.float32)

        if self._backend == "cohere":
            response = self._model.embed(texts=texts, model=self.model_name, input_type="search_document")
            return np.asarray(response.embeddings, dtype=np.float32)

        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)


class LocalVectorStore:
    def __init__(self, storage_dir: str | Path = "data/index"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_dir / "resume_index.json"
        self.embedding_path = self.storage_dir / "embeddings.npy"
        self._chroma_client = None
        self._chroma_collection = None
        self._chroma_enabled = False
        self.records: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.backend_name = "local"
        try:
            import chromadb

            self._chroma_client = chromadb.PersistentClient(path=str(self.storage_dir / "chroma"))
            self._chroma_collection = self._chroma_client.get_or_create_collection("resumes")
            self._chroma_enabled = True
            self.backend_name = "chroma"
        except Exception:
            self._chroma_enabled = False

    def _reset_chroma_collection(self) -> None:
        if not self._chroma_enabled or self._chroma_client is None:
            return
        try:
            self._chroma_client.delete_collection("resumes")
        except Exception:
            pass
        self._chroma_collection = self._chroma_client.get_or_create_collection("resumes")

    def save(self) -> None:
        with self.index_path.open("w", encoding="utf-8") as handle:
            json.dump(self.records, handle, ensure_ascii=False, indent=2)
        if self.embeddings is not None:
            np.save(self.embedding_path, self.embeddings)
        if self._chroma_enabled and self._chroma_collection is not None and self.embeddings is not None:
            self._reset_chroma_collection()
            ids = [f"resume-{index}" for index in range(len(self.records))]
            documents = [record["chunk_text"] for record in self.records]
            metadatas = [
                {
                    "candidate_name": record["candidate_name"],
                    "resume_path": record["resume_path"],
                    "section": record["section"],
                    "chunk_index": int(record["chunk_index"]),
                    "skills": ", ".join(record.get("skills", [])),
                    "experience_years": record.get("experience_years"),
                    "education": " | ".join(record.get("education", [])),
                }
                for record in self.records
            ]
            self._chroma_collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=self.embeddings.tolist())

    def load(self) -> None:
        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8") as handle:
                self.records = json.load(handle)
        if self.embedding_path.exists():
            self.embeddings = np.load(self.embedding_path)
        if self._chroma_enabled and self._chroma_collection is not None:
            result = self._chroma_collection.get(include=["documents", "metadatas", "embeddings"])
            if result.get("ids"):
                self.records = []
                documents = result.get("documents") or []
                metadatas = result.get("metadatas") or []
                embeddings = result.get("embeddings")
                if embeddings is None:
                    embeddings = []
                rebuilt_embeddings: list[list[float]] = []
                for index, metadata in enumerate(metadatas):
                    record = {
                        "candidate_name": metadata.get("candidate_name", ""),
                        "resume_path": metadata.get("resume_path", ""),
                        "section": metadata.get("section", "summary"),
                        "chunk_text": documents[index] if index < len(documents) else "",
                        "chunk_index": int(metadata.get("chunk_index", index)),
                        "skills": [skill.strip() for skill in str(metadata.get("skills", "")).split(",") if skill.strip()],
                        "experience_years": metadata.get("experience_years"),
                        "education": [item.strip() for item in str(metadata.get("education", "")).split("|") if item.strip()],
                        "source_text": documents[index] if index < len(documents) else "",
                    }
                    self.records.append(record)
                    if index < len(embeddings):
                        rebuilt_embeddings.append(embeddings[index])
                if rebuilt_embeddings:
                    self.embeddings = np.asarray(rebuilt_embeddings, dtype=np.float32)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        if self.embeddings is None or not len(self.records):
            return []
        if self._chroma_enabled and self._chroma_collection is not None:
            result = self._chroma_collection.query(query_embeddings=[np.asarray(query_vector, dtype=np.float32).tolist()], n_results=top_k)
            ids = result.get("ids", [[]])[0]
            distances = result.get("distances", [[]])[0]
            lookup = {f"resume-{index}": index for index in range(len(self.records))}
            ranked: list[tuple[int, float]] = []
            for item_id, distance in zip(ids, distances):
                if item_id in lookup:
                    ranked.append((lookup[item_id], float(1.0 - distance)))
            if ranked:
                return ranked
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        matrix = np.asarray(self.embeddings, dtype=np.float32)
        denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        denom = np.maximum(denom, 1e-8)
        scores = (matrix @ query.T).reshape(-1) / denom
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(index), float(scores[index])) for index in ranked]


def iter_resume_files(resume_root: str | Path) -> Iterable[Path]:
    root = Path(resume_root)
    if not root.exists():
        return []
    return sorted([path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}])


def read_resume_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return path.read_text(encoding="utf-8", errors="ignore")
    return path.read_text(encoding="utf-8", errors="ignore")


def candidate_name_from_text(text: str, path: Path) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not SECTION_PATTERN.match(stripped):
            if len(stripped.split()) <= 5 and not stripped.lower().startswith(("summary", "profile")):
                return stripped.replace("Name:", "").strip()
            break
    return path.stem.replace("_", " ").title()


def split_into_section_blocks(text: str) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    current_section = "summary"
    current_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if SECTION_PATTERN.match(line):
            if current_lines:
                blocks.append((current_section, current_lines))
            current_section = line.rstrip(":").lower()
            current_lines = []
            continue
        if line:
            current_lines.append(line)
    if current_lines:
        blocks.append((current_section, current_lines))
    return blocks


def split_section_to_chunks(section: str, lines: list[str], max_chars: int = 700) -> list[str]:
    chunks: list[str] = []
    buffer: list[str] = []
    current_length = 0
    for line in lines:
        if current_length + len(line) + 1 > max_chars and buffer:
            chunks.append(" ".join(buffer))
            buffer = []
            current_length = 0
        buffer.append(line)
        current_length += len(line) + 1
    if buffer:
        chunks.append(" ".join(buffer))
    return [f"{section.title()}: {chunk}" for chunk in chunks]


def extract_skills(text: str) -> list[str]:
    lower = text.lower()
    found = {hint.title() for hint in SKILL_HINTS if hint in lower}
    return sorted(found)


def extract_experience_years(text: str) -> float | None:
    matches = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\+?\s+years", text, flags=re.IGNORECASE)]
    if matches:
        return max(matches)
    match = re.search(r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def extract_education(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    education_lines: list[str] = []
    capture = False
    for line in lines:
        if re.match(r"^education\b", line, flags=re.IGNORECASE):
            capture = True
            continue
        if capture and SECTION_PATTERN.match(line):
            break
        if capture and line:
            education_lines.append(line)
    if not education_lines:
        degree_matches = re.findall(r"(b\.?s\.?|m\.?s\.?|ph\.?d\.?|mba|bachelor|master|associate)[^.;\n]*", text, flags=re.IGNORECASE)
        education_lines = [match.strip() for match in degree_matches]
    return education_lines[:4]


def chunk_resume_document(text: str, path: Path) -> list[ResumeChunk]:
    candidate_name = candidate_name_from_text(text, path)
    skills = extract_skills(text)
    experience_years = extract_experience_years(text)
    education = extract_education(text)
    chunks: list[ResumeChunk] = []
    blocks = split_into_section_blocks(text)
    chunk_index = 0
    for section, lines in blocks:
        for chunk_text in split_section_to_chunks(section, lines):
            chunks.append(
                ResumeChunk(
                    candidate_name=candidate_name,
                    resume_path=str(path.as_posix()),
                    section=section,
                    chunk_text=chunk_text,
                    chunk_index=chunk_index,
                    skills=skills,
                    experience_years=experience_years,
                    education=education,
                    source_text=text,
                )
            )
            chunk_index += 1
    if not chunks:
        chunks.append(
            ResumeChunk(
                candidate_name=candidate_name,
                resume_path=str(path.as_posix()),
                section="summary",
                chunk_text=text[:700],
                chunk_index=0,
                skills=skills,
                experience_years=experience_years,
                education=education,
                source_text=text,
            )
        )
    return chunks


def build_resume_index(
    resume_root: str | Path = "data/resumes",
    storage_dir: str | Path = "data/index",
    provider: str = "hf",
    model_name: str | None = None,
) -> LocalVectorStore:
    resume_root = Path(resume_root)
    store = LocalVectorStore(storage_dir=storage_dir)
    texts: list[str] = []
    records: list[dict[str, Any]] = []

    for path in iter_resume_files(resume_root):
        text = read_resume_text(path)
        for chunk in chunk_resume_document(text, path):
            record = asdict(chunk)
            records.append(record)
            texts.append(chunk.chunk_text)

    if not texts:
        raise FileNotFoundError(f"No resume files found under {resume_root}")

    backend = EmbeddingBackend(provider=provider, model_name=model_name)
    backend.fit(texts)
    embeddings = backend.encode(texts)

    store.records = records
    store.embeddings = embeddings
    store.save()

    metadata = {
        "provider": provider,
        "model_name": backend.model_name,
        "backend": backend._backend,
        "records": len(records),
        "resume_root": str(resume_root),
    }
    with (store.storage_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return store


def load_resume_index(storage_dir: str | Path = "data/index") -> LocalVectorStore:
    store = LocalVectorStore(storage_dir=storage_dir)
    store.load()
    return store


def keyword_match_score(text: str, query: str) -> float:
    text_lower = text.lower()
    query_terms = [term for term in re.findall(r"[A-Za-z0-9+#.-]+", query.lower()) if len(term) > 2]
    if not query_terms:
        return 0.0
    hits = sum(1 for term in query_terms if term in text_lower)
    return hits / max(len(query_terms), 1)


def parse_must_have_requirements(job_description: str) -> dict[str, Any]:
    requirements: dict[str, Any] = {"skills": [], "years": None}
    years_match = re.search(r"(\d+(?:\.\d+)?)\+?\s+years?.{0,20}?([A-Za-z][A-Za-z0-9+#./ -]{1,40})", job_description, flags=re.IGNORECASE)
    if years_match:
        requirements["years"] = {"minimum": float(years_match.group(1)), "skill": years_match.group(2).strip()}
    for line in job_description.splitlines():
        if re.search(r"must have|required|mandatory|need", line, flags=re.IGNORECASE):
            requirements["skills"].extend(extract_skills(line))
    if not requirements["skills"]:
        requirements["skills"] = extract_skills(job_description)[:6]
    return requirements


def cosine_similarity_scores(matrix: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    numerator = matrix @ query.T
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True) * np.linalg.norm(query)
    denominator = np.maximum(denominator, 1e-8)
    return (numerator / denominator).reshape(-1)


def summarize_excerpts(text: str, query_terms: list[str], limit: int = 2) -> list[str]:
    excerpts: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        lower = sentence.lower()
        if any(term.lower() in lower for term in query_terms):
            excerpts.append(sentence.strip())
        if len(excerpts) >= limit:
            break
    if not excerpts:
        excerpts = [text[:250].strip()]
    return excerpts


def build_match_reasoning(candidate: dict[str, Any], semantic_score: float, keyword_score: float, matched_skills: list[str]) -> str:
    years = candidate.get("experience_years")
    section = candidate.get("section", "resume")
    reasoning_parts = [
        f"Strong semantic overlap in {section} section",
        f"keyword boost {keyword_score:.2f}",
        f"matched skills: {', '.join(matched_skills) if matched_skills else 'none'}",
    ]
    if years is not None:
        reasoning_parts.append(f"experience noted at {years} years")
    reasoning_parts.append(f"semantic score {semantic_score:.2f}")
    return "; ".join(reasoning_parts)


def apply_must_have_filter(candidate: dict[str, Any], requirements: dict[str, Any]) -> bool:
    years_requirement = requirements.get("years")
    if years_requirement:
        candidate_years = candidate.get("experience_years")
        if candidate_years is None or float(candidate_years) < float(years_requirement["minimum"]):
            skill = str(years_requirement["skill"]).lower()
            if skill and skill not in candidate.get("source_text", "").lower():
                return False
    required_skills = [skill.lower() for skill in requirements.get("skills", [])]
    if required_skills:
        candidate_skills = {skill.lower() for skill in candidate.get("skills", [])}
        if not any(skill in candidate_skills or skill in candidate.get("source_text", "").lower() for skill in required_skills):
            return False
    return True


def match_job_description(
    job_description: str,
    storage_dir: str | Path = "data/index",
    top_k: int = 10,
    provider: str = "hf",
    model_name: str | None = None,
) -> dict[str, Any]:
    store = load_resume_index(storage_dir=storage_dir)
    if not store.records or store.embeddings is None:
        raise FileNotFoundError("Resume index is empty. Build it before matching.")

    backend = EmbeddingBackend(provider=provider, model_name=model_name)
    if provider == "tfidf":
        backend.fit([record["chunk_text"] for record in store.records])
    else:
        backend.fit([job_description] + [record["chunk_text"] for record in store.records[: min(10, len(store.records))]])
    query_vector = backend.encode([job_description])[0]
    semantic_scores = cosine_similarity_scores(store.embeddings, query_vector)

    requirements = parse_must_have_requirements(job_description)
    query_terms = sorted({term for term in re.findall(r"[A-Za-z0-9+#.-]+", job_description) if len(term) > 2})

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for index, record in enumerate(store.records):
        key = (record["candidate_name"], record["resume_path"])
        semantic_score = float(semantic_scores[index])
        keyword_score = keyword_match_score(record["chunk_text"], job_description)
        if key not in grouped or semantic_score > grouped[key]["score"]:
            grouped[key] = {
                **record,
                "score": semantic_score,
                "keyword_score": keyword_score,
            }

    ranked_candidates = sorted(
        grouped.values(),
        key=lambda item: (0.75 * item["score"] + 0.25 * item["keyword_score"]),
        reverse=True,
    )

    results: list[dict[str, Any]] = []
    for candidate in ranked_candidates:
        if not apply_must_have_filter(candidate, requirements):
            continue
        matched_skills = [skill for skill in candidate.get("skills", []) if skill.lower() in job_description.lower()]
        relevant_excerpts = summarize_excerpts(candidate.get("source_text", ""), query_terms + matched_skills)
        semantic_score = float(candidate["score"])
        keyword_score = float(candidate["keyword_score"])
        final_score = min(100.0, round((0.8 * semantic_score + 0.2 * keyword_score) * 100 + len(matched_skills) * 3, 2))
        results.append(
            {
                "candidate_name": candidate["candidate_name"],
                "resume_path": candidate["resume_path"],
                "match_score": final_score,
                "matched_skills": matched_skills,
                "relevant_excerpts": relevant_excerpts,
                "reasoning": build_match_reasoning(candidate, semantic_score, keyword_score, matched_skills),
            }
        )
        if len(results) >= top_k:
            break

    return {"job_description": job_description, "top_matches": results}


def evaluate_retrieval(
    job_descriptions: list[str],
    expected_candidate_names: list[str],
    storage_dir: str | Path = "data/index",
    provider: str = "hf",
) -> dict[str, Any]:
    correct = 0
    latencies_ms: list[float] = []
    for job_description, expected_name in zip(job_descriptions, expected_candidate_names):
        import time

        start = time.perf_counter()
        result = match_job_description(job_description, storage_dir=storage_dir, top_k=1, provider=provider)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        top_match = result["top_matches"][0]["candidate_name"] if result["top_matches"] else None
        if top_match == expected_name:
            correct += 1
    return {
        "retrieval_accuracy": correct / max(len(job_descriptions), 1),
        "avg_latency_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a resume RAG index from local files.")
    parser.add_argument("--resume-root", default="data/resumes")
    parser.add_argument("--storage-dir", default="data/index")
    parser.add_argument("--provider", default="hf")
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    store = build_resume_index(
        resume_root=args.resume_root,
        storage_dir=args.storage_dir,
        provider=args.provider,
        model_name=args.model_name,
    )
    print(json.dumps({"records": len(store.records), "storage_dir": str(store.storage_dir)}, indent=2))


if __name__ == "__main__":
    main()