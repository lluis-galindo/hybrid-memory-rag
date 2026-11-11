import json
import re
import sys
from pathlib import Path

import numpy as np

from src.corpus import episode_ids, load_scenes
from src.embeddings import Embedder
from src.memory import build_scene_index
from src.utils import SqliteCache, read_jsonl

EMBEDDERS = {
    "bge-m3": ("", ""),
    "nomic-embed-text": ("search_query: ", "search_document: "),
}
KS = (1, 5, 10, 20)
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def recap_queries(path: Path, episode_index: dict[str, int]) -> list[tuple[str, int]]:
    queries = []
    for row in read_jsonl(path):
        for sentence in SENTENCE.split(row["recap"]):
            if len(sentence.split()) >= 8:
                queries.append((sentence.strip(), episode_index[row["episode_id"]]))
    return queries


def evaluate(model: str, scenes, queries, cache_dir: Path) -> dict:
    query_prefix, document_prefix = EMBEDDERS[model]
    embedder = Embedder(model, cache=SqliteCache(cache_dir / "embeddings.sqlite"), query_prefix=query_prefix, document_prefix=document_prefix)
    store = build_scene_index(scenes, embedder)
    hits_at = {k: 0 for k in KS}
    reciprocal = []
    for text, target in queries:
        ranked = store.search(embedder.embed_query(text), max(KS), max_episode_index=target)
        ranks = [i for i, hit in enumerate(ranked) if hit.record.episode_index == target]
        reciprocal.append(1.0 / (ranks[0] + 1) if ranks else 0.0)
        for k in KS:
            hits_at[k] += bool(ranks and ranks[0] < k)
    return {"embedder": model, "queries": len(queries), **{f"episode_recall@{k}": round(v / len(queries), 3) for k, v in hits_at.items()}, "mrr@20": round(float(np.mean(reciprocal)), 3)}


def main() -> None:
    scenes = load_scenes(Path(sys.argv[1]))
    episode_index = {episode: i for i, episode in enumerate(episode_ids(scenes))}
    queries = recap_queries(Path(sys.argv[2]), episode_index)
    results = [evaluate(model, scenes, queries, Path(".cache")) for model in sys.argv[3].split(",")]
    print(json.dumps(results, indent=2))
    Path(sys.argv[4]).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
