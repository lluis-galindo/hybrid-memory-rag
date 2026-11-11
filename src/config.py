from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMConfig(Section):
    model: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    num_ctx: int = 8192
    think: bool | None = False
    seed: int = 7
    timeout_s: float = 900.0


class EmbeddingConfig(Section):
    model: str
    base_url: str = "http://localhost:11434"
    batch_size: int = 32
    query_prefix: str = ""
    document_prefix: str = ""
    timeout_s: float = 300.0


class SegmentationConfig(Section):
    min_scene_tokens: int = 40
    max_scene_tokens: int = 700
    location_breaks: bool = False


class RetrievalConfig(Section):
    context_tokens: int = 4000
    top_k: int = 20
    graph_tokens: int = 1000
    memory_tokens: int = 600
    memory_top_k: int = 10
    max_graph_entities: int = 6
    long_context_tokens: int = 28000
    long_context_num_ctx: int = 32768


class ShortTermConfig(Section):
    budget_tokens: int = 1500
    keep_recent_turns: int = 4


class AgentConfig(Section):
    max_search_rounds: int = 2
    search_top_k: int = 10
    abstain_when_unsupported: bool = True


class PathsConfig(Section):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    runs_dir: Path = Path("runs")
    cache_dir: Path = Path(".cache")


class Config(Section):
    series: str
    llm: LLMConfig
    judge: LLMConfig
    embedding: EmbeddingConfig
    segmentation: SegmentationConfig = SegmentationConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    short_term: ShortTermConfig = ShortTermConfig()
    agents: AgentConfig = AgentConfig()
    paths: PathsConfig = PathsConfig()

    @property
    def series_dir(self) -> Path:
        return self.paths.processed_dir / self.series

    @property
    def scenes_path(self) -> Path:
        return self.series_dir / "scenes.jsonl"

    @property
    def scene_index_dir(self) -> Path:
        return self.series_dir / "scene_index"

    @property
    def memory_index_dir(self) -> Path:
        return self.series_dir / "memory_index"

    @property
    def graph_path(self) -> Path:
        return self.series_dir / "graph.json"

    @property
    def consolidation_log_path(self) -> Path:
        return self.series_dir / "consolidation_log.jsonl"


def load_config(path: Path) -> Config:
    return Config.model_validate(yaml.safe_load(path.read_text()))
