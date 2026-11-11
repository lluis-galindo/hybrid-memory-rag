from functools import cached_property

from .agents import MultiAgentSystem
from .config import Config, LLMConfig
from .corpus import Scene, episode_ids, load_scenes
from .embeddings import Embedder
from .graph import TemporalKnowledgeGraph
from .llm import LLM
from .memory import ShortTermMemory
from .retrieval import HybridLongTermMemory
from .systems import HybridMemorySystem, LongContextBaseline, QASystem, VectorRAGBaseline
from .utils import SqliteCache
from .vector_store import VectorStore

SYSTEM_NAMES = ("B0", "B1", "C1", "C2")


class Workspace:
    def __init__(self, config: Config):
        self.config = config

    @cached_property
    def llm_cache(self) -> SqliteCache:
        return SqliteCache(self.config.paths.cache_dir / "llm.sqlite")

    def _build_llm(self, config: LLMConfig) -> LLM:
        options = {"temperature": config.temperature, "num_ctx": config.num_ctx, "seed": config.seed}
        return LLM(config.model, config.base_url, options, config.think, config.timeout_s, self.llm_cache)

    @cached_property
    def llm(self) -> LLM:
        return self._build_llm(self.config.llm)

    @cached_property
    def judge_llm(self) -> LLM:
        return self._build_llm(self.config.judge)

    @cached_property
    def embedder(self) -> Embedder:
        config = self.config.embedding
        cache = SqliteCache(self.config.paths.cache_dir / "embeddings.sqlite")
        return Embedder(config.model, config.base_url, config.timeout_s, cache, config.batch_size, config.query_prefix, config.document_prefix)

    @cached_property
    def scenes(self) -> list[Scene]:
        if not self.config.scenes_path.exists():
            raise FileNotFoundError(f"{self.config.scenes_path} not found; run the ingest command first")
        return load_scenes(self.config.scenes_path)

    @cached_property
    def episodes(self) -> list[str]:
        return episode_ids(self.scenes)

    @cached_property
    def episode_index(self) -> dict[str, int]:
        return {episode: index for index, episode in enumerate(self.episodes)}

    @cached_property
    def scene_index(self) -> VectorStore:
        return VectorStore.load(self.config.scene_index_dir, self.embedder.name)

    @cached_property
    def memory_index(self) -> VectorStore:
        return VectorStore.load(self.config.memory_index_dir, self.embedder.name)

    @cached_property
    def graph(self) -> TemporalKnowledgeGraph:
        return TemporalKnowledgeGraph.load(self.config.graph_path)

    @cached_property
    def long_term(self) -> HybridLongTermMemory:
        return HybridLongTermMemory(self.scene_index, self.memory_index, self.graph, self.embedder, self.config.retrieval)

    def system(self, name: str) -> QASystem:
        retrieval = self.config.retrieval
        if name == "B0":
            return LongContextBaseline(self.llm, self.scenes, retrieval.long_context_tokens, retrieval.long_context_num_ctx)
        if name == "B1":
            return VectorRAGBaseline(self.llm, self.scene_index, self.embedder, retrieval.top_k, retrieval.context_tokens)
        if name == "C1":
            return HybridMemorySystem(self.llm, self.long_term)
        if name == "C2":
            short_term = ShortTermMemory(self.config.short_term.budget_tokens, self.config.short_term.keep_recent_turns, self.llm)
            return MultiAgentSystem(self.llm, self.long_term, short_term, self.config.agents)
        raise ValueError(f"Unknown system {name!r}; choose from {SYSTEM_NAMES}")
