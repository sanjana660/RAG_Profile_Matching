from __future__ import annotations

import argparse
import json
from pathlib import Path

from resume_rag import build_resume_index, match_job_description


def load_text(source: str | Path) -> str:
    path = Path(source)
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return str(source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Match a job description against resume chunks.")
    parser.add_argument("job_description", help="Path to a job description file or raw job description text")
    parser.add_argument("--storage-dir", default="data/index")
    parser.add_argument("--resume-root", default="data/resumes")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--provider", default="hf")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--build-index", action="store_true")
    args = parser.parse_args()

    if args.build_index or not (Path(args.storage_dir) / "resume_index.json").exists():
        build_resume_index(
            resume_root=args.resume_root,
            storage_dir=args.storage_dir,
            provider=args.provider,
            model_name=args.model_name,
        )

    job_description = load_text(args.job_description)
    result = match_job_description(
        job_description=job_description,
        storage_dir=args.storage_dir,
        top_k=args.top_k,
        provider=args.provider,
        model_name=args.model_name,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()