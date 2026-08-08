# RAG Based Profile Matching

This project implements a resume RAG pipeline, semantic retrieval, and job-to-candidate ranking.

## What is included

- `resume_rag.py` for resume loading, section-aware chunking, metadata extraction, embeddings, and local vector indexing.
- `job_matcher.py` for hybrid semantic + keyword matching and JSON output.
- `data/resumes/` with 30+ diverse resume documents.
- `data/jobs/` with 5+ job descriptions.
- `notebooks/rag_experiment.ipynb` for experimentation and metrics.

## Run

### Step-by-step commands (Windows PowerShell)

```powershell
cd c:\Users\User\Airtribe\RAG_Based_Profile_matching
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python resume_rag.py --resume-root data/resumes --storage-dir data/index
python job_matcher.py data/jobs/data_scientist.txt --storage-dir data/index --top-k 10
```

### Optional: try a different job file

```powershell
python job_matcher.py data/jobs/data_engineer.txt --storage-dir data/index --top-k 10
python job_matcher.py data/jobs/ml_engineer.txt --storage-dir data/index --top-k 10
```

## Output

The matcher returns a JSON object shaped like the assignment example, including candidate names, scores, excerpts, and reasoning.

## Notes

- The embedding backend defaults to Hugging Face `sentence-transformers` when available.
- OpenAI and Cohere backends are supported through the same interface if API credentials are configured.
- If external embedding libraries are unavailable, the code falls back to a TF-IDF embedding model so the project still runs locally.