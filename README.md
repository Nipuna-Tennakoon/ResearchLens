# ResearchLens

A command line RAG application for research papers. Point it at a folder of PDFs,
and ask questions that are answered from those papers with the source passages shown.

It has two components:

- **Data ingestion** — read PDFs, clean the text, chunk it, embed it and store the
  vectors in Milvus.
- **Retrieval** — embed a question, search the vectors, rerank the candidates with
  a cross-encoder and answer from the best ones with an LLM.

## Setup

```bash
uv sync                      # install dependencies
cp .env.example .env         # then add your OPENAI_API_KEY
docker compose up -d         # start Milvus on localhost:19530
```

Embeddings run locally by default with the BERT sentence-transformer
`all-MiniLM-L6-v2` (384 dimensions), downloaded on first use. OpenAI is only
called for title extraction during ingestion and for writing the answer.

The model takes a while to load, so the CLI starts loading it on a background
thread at start-up and ingestion and retrieval share that one instance. Commands
that never embed anything (`status`, `reset`) do not load it at all.

It is downloaded once into the standard HuggingFace cache
(`~/.cache/huggingface/hub`, or `$HF_HOME`) and loaded from there on every later
run — ResearchLens checks the cache first and only contacts the Hub when
something is genuinely missing. Skipping that check costs about 90 seconds per
start-up in revalidation requests.

## Usage

Run without arguments for the interactive menu. The wordmark animates while the
models load in the background, then the menu appears with everything warm:

```bash
uv run researchlens
```

```
                     R e s e a r c h L e n s
                retrieval-augmented research assistant
```

Pass `--no-splash` to skip the animation. It is skipped automatically when output
is not a terminal, so piping and scripting are unaffected.

```
╭──────────────────────── ResearchLens v0.1.0 ─────────────────────────╮
│ 1  Data ingestion — load PDFs from a folder into the vector database │
│ 2  Retrieval — ask questions about the indexed papers                │
│ 3  Status — inspect the vector database                              │
│ 4  Exit                                                              │
╰──────────────────────────────────────────────────────────────────────╯
```

Or call the commands directly, which is what you want in a script or a job:

```bash
uv run researchlens ingest ./data                       # ingest a folder of PDFs
EXTRACT_TITLES=false uv run researchlens ingest ./data  # faster, no OpenAI calls
uv run researchlens ask "What parameters does SARIMA use?"
uv run researchlens ask                                 # interactive Q&A session
uv run researchlens ask "..." --sources                  # also list the chunks used
uv run researchlens status                              # what is indexed
uv run researchlens reset --yes                         # drop the collection
uv run researchlens --verbose ingest ./data             # INFO level logs
```

`ingest` searches the folder recursively, and re-ingesting a folder replaces the
existing rows instead of duplicating them. A PDF that fails to parse is reported at
the end of the run without stopping the other files.

## How retrieval ranks

Vector search scores the question and each chunk separately, which is fast enough
to run over a whole collection but only approximate. A cross-encoder reads the
question and a chunk *together* and scores that pair directly — much more
accurate, far too slow to run over everything.

ResearchLens uses both: Milvus returns `SEARCH_LIMIT` candidates, the
`BAAI/bge-reranker-base` cross-encoder rescores each one against the question,
and the best `TOP_K` become the LLM's context. Pass `--sources` to print those
chunks with both scores, so you can see where the two rankings disagree.

Reranking costs roughly 0.4s per candidate on CPU. Set `RERANK=false` to skip it.

## Configuration

Every setting has a working default and is overridable through the environment or
`.env` — see [.env.example](.env.example) for the full list. The ones you are most
likely to change:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Needed for title extraction and the answer model, and for `EMBED_PROVIDER=openai`. Local ingestion with `EXTRACT_TITLES=false` needs no key |
| `MILVUS_URI` | `http://localhost:19530` | Milvus address |
| `MILVUS_COLLECTION` | `paper_chunks_2` | Collection to read and write |
| `EMBED_PROVIDER` | `huggingface` | `huggingface` for a local BERT model, `openai` for the API |
| `EMBED_MODEL` / `EMBED_DIM` | `all-MiniLM-L6-v2` / `384` | Embedding model and its dimension. Both default to the right pair for the provider, so setting `EMBED_PROVIDER=openai` alone gives `text-embedding-3-small` / `1536` |
| `LLM_MODEL` | `gpt-4o-mini` | Model that writes the answer |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `150` / `10` | Sentence splitter settings |
| `SEARCH_LIMIT` | `10` | Candidates fetched from Milvus and handed to the reranker |
| `TOP_K` | `3` | Chunks kept after reranking and passed to the LLM as context |
| `RERANK` | `true` | Set `false` to skip the cross-encoder and rank by vector similarity alone |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder used for reranking |
| `TEMPERATURE` | `0` | Sampling temperature for the answer model |
| `EXTRACT_TITLES` | `true` | Derive a document title per chunk with an LLM. Set `false` for faster, cheaper ingestion — the file name is used instead |

Ingestion and retrieval always share one embedding model, so query vectors match
the stored ones. Changing `EMBED_MODEL` invalidates what is already indexed —
ResearchLens detects the mismatch and refuses to read or write the collection,
rather than returning nonsense. Point `MILVUS_COLLECTION` somewhere else, or run
`researchlens reset` and ingest again.

Fully local, zero-cost ingestion:

```bash
EXTRACT_TITLES=false uv run researchlens ingest ./data
```

The file name is used as the document title instead of an LLM-generated one.

## Layout

```
src/researchlens/
  cli.py             Typer commands and the interactive menu
  splash.py          Animated start-up wordmark
  config.py          Settings, loaded from the environment
  embeddings.py      Loads the embedding model once, in the background, and shares it
  ingestion.py       PDFs -> cleaned text -> chunks -> embeddings
  normalization.py   Unicode, hyphenation and whitespace clean-up
  store.py           Milvus schema, upserts and vector search
  retrieval.py       Embed, search, rerank, prompt the LLM
  reranking.py       Cross-encoder reranking of the candidate chunks
  model_loading.py   HuggingFace cache lookup and background model loading
notebooks/           The experiments this CLI was built from
tests/               Pipeline tests using in-memory fakes
```

## Tests

```bash
uv run pytest
```

The suite runs the real PDF parsing and chunking pipeline, with in-memory fakes in
place of Milvus and the embedding/answer models, so it needs neither network
access nor a key. Tests that download the real BERT model are marked `slow` and
skipped by default:

```bash
uv run pytest -m slow
```
