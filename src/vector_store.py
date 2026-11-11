import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .utils import read_jsonl, write_jsonl


@dataclass
class VectorRecord:
    id: str
    text: str
    kind: str
    episode_id: str
    episode_index: int
    scene_index: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def order_key(self) -> tuple[int, int]:
        return (self.episode_index, self.scene_index)


@dataclass
class Hit:
    record: VectorRecord
    score: float


class VectorStore:
    def __init__(self, records: list[VectorRecord], vectors: np.ndarray, embedder_name: str):
        self.records = records
        self.vectors = vectors.astype(np.float32)
        self.embedder_name = embedder_name

    def __len__(self) -> int:
        return len(self.records)

    def search(self, query: np.ndarray, k: int, max_episode_index: int | None = None, exclude_ids: Iterable[str] = ()) -> list[Hit]:
        excluded = set(exclude_ids)
        limit = float("inf") if max_episode_index is None else max_episode_index
        candidates = [i for i, r in enumerate(self.records) if r.episode_index <= limit and r.id not in excluded]
        if not candidates:
            return []
        scores = self.vectors @ query.astype(np.float32)
        ranked = np.asarray(candidates)[np.argsort(-scores[candidates], kind="stable")[:k]]
        return [Hit(self.records[i], float(scores[i])) for i in ranked]

    def delete(self, predicate: Callable[[VectorRecord], bool]) -> int:
        keep = [i for i, record in enumerate(self.records) if not predicate(record)]
        removed = len(self.records) - len(keep)
        self.records = [self.records[i] for i in keep]
        self.vectors = self.vectors[keep]
        return removed

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        write_jsonl(directory / "records.jsonl", (asdict(record) for record in self.records))
        (directory / "meta.json").write_text(json.dumps({"dim": self.vectors.shape[1], "embedder": self.embedder_name}))

    @classmethod
    def load(cls, directory: Path, expected_embedder: str) -> "VectorStore":
        embedder = json.loads((directory / "meta.json").read_text())["embedder"]
        if embedder != expected_embedder:
            raise ValueError(f"Index at {directory} was built with {embedder}, current embedder is {expected_embedder}")
        records = [VectorRecord(**row) for row in read_jsonl(directory / "records.jsonl")]
        return cls(records, np.load(directory / "vectors.npy"), embedder)
