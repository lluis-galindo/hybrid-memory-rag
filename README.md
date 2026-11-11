# Hybrid Short- and Long-Term Memory for Multi-Agent RAG Systems

Master's thesis (TFM) by **Lluís Galindo Vandrell**: narrative understanding over TV series subtitles and transcripts.
The prototype answers questions about a series using only what has happened up to a given episode, and everything
runs locally with Ollama. The development log is in `docs/TFM_implementation.md`.

It compares four configurations: B0 (long context), B1 (vector RAG), C1 (vector store + temporal knowledge graph) and
C2 (C1 + critic agent, built with LangGraph).

## Structure

- `src/`: the code (corpus, memory, retrieval, systems, agents, questions and evaluation); `src/cli.py` is the entry point.
- `configs/alias.yaml`: models and parameters.
- `scripts/`: the experiments behind the tables of the report.
- `data/`: corpus, questions and a small fictional sample used by the tests.
- `docs/`: thesis chapter and experiment results.
- `tests/`: unit and end-to-end tests, which need no models.

## Data

The corpus is *Alias*, seasons 1 and 2 (44 episodes, about 270,000 words, 1,052 scenes), taken from the ForeverDreaming
subset of SummScreen (Chen et al., ACL 2022). The transcripts are not in the repository (`data/raw/` is ignored), so
download the untokenized SummScreen release first. The evaluation questions are generated from the human-written
episode recaps and then reviewed by hand.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
ollama pull qwen3:14b && ollama pull gemma4:12b-it-qat && ollama pull bge-m3
```

From the repository root:

```bash
python -m src.cli prepare-summscreen --source <SummScreen_raw/fd> --show Alias --seasons 1,2
python -m src.cli ingest
python -m src.cli index
python -m src.cli build-memory
python -m src.cli generate-questions --recaps data/raw/alias_recaps.jsonl --out data/questions/alias/candidates.jsonl
python -m src.cli review --candidates data/questions/alias/candidates.jsonl --out data/questions/alias/questions.jsonl
python -m src.cli run --questions data/questions/alias/questions.jsonl --run-id main
python -m src.cli judge --questions data/questions/alias/questions.jsonl --run-id main
python -m src.cli report --questions data/questions/alias/questions.jsonl --run-id main
```

Results are written to `runs/<run-id>/`. Use `python -m src.cli --help` to see the other commands.
