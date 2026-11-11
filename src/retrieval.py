import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .config import RetrievalConfig
from .embeddings import Embedder
from .graph import Event, Fact, TemporalKnowledgeGraph, normalize_name
from .utils import count_tokens
from .vector_store import Hit, VectorStore

SCENES_HEADER = "## Transcript scenes"
GRAPH_HEADER = "## Knowledge graph (facts and events known so far)"
MEMORIES_HEADER = "## Episode memories"
SPEAKER_SEED_HITS = 3

STOPWORDS = frozenset(
    "a an the and or but of to in on at for with from by is are was were be been do does did what who whom whose "
    "which when where why how that this these those it its he she they them his her their there than then has have had "
    "about into after before during over between first last ever".split()
)
WORD = re.compile(r"[a-z0-9']+")


def content_words(text: str) -> set[str]:
    return {word for word in WORD.findall(text.lower()) if word not in STOPWORDS and len(word) > 1}


def lexical_overlap(query_words: set[str], text: str) -> float:
    return len(query_words & content_words(text)) / len(query_words) if query_words else 0.0


def render_scene(scene_id: str, episode_id: str, text: str) -> str:
    return f"[{scene_id} | {episode_id}]\n{text}"


def select_within(lines: Sequence[str], budget: int) -> list[int]:
    chosen: list[int] = []
    used = 0
    for position, line in enumerate(lines):
        size = count_tokens(line) + 1
        if used + size <= budget:
            chosen.append(position)
            used += size
    return chosen


@dataclass
class ContextBundle:
    text: str
    tokens: int
    context_scene_ids: list[str]
    ranked_scene_ids: list[str]
    sections: dict[str, int] = field(default_factory=dict)


def compose_context(
    scene_hits: Sequence[Hit],
    budget: int,
    graph_lines: Sequence[str] = (),
    memory_lines: Sequence[str] = (),
    graph_budget: int = 0,
    memory_budget: int = 0,
) -> ContextBundle:
    parts: list[str] = []
    sections: dict[str, int] = {}
    used = 0
    for header, lines, section_budget, name in (
        (GRAPH_HEADER, graph_lines, graph_budget, "graph"),
        (MEMORIES_HEADER, memory_lines, memory_budget, "memories"),
    ):
        allowed = min(section_budget, budget - used) - count_tokens(header) - 1
        chosen = [lines[i] for i in select_within(lines, allowed)]
        if chosen:
            block = header + "\n" + "\n".join(chosen)
            parts.append(block)
            used += count_tokens(block) + 2
            sections[name] = len(chosen)
    unique: dict[str, Hit] = {}
    for hit in scene_hits:
        unique.setdefault(hit.record.id, hit)
    hits = list(unique.values())
    blocks = [render_scene(h.record.id, h.record.episode_id, h.record.text) for h in hits]
    remaining = budget - used - count_tokens(SCENES_HEADER) - 1
    chosen_positions = select_within([block + "\n" for block in blocks], remaining)
    chosen_hits = sorted((hits[i] for i in chosen_positions), key=lambda h: h.record.order_key)
    if chosen_hits:
        parts.append(SCENES_HEADER + "\n" + "\n\n".join(render_scene(h.record.id, h.record.episode_id, h.record.text) for h in chosen_hits))
        sections["scenes"] = len(chosen_hits)
    text = "\n\n".join(parts)
    return ContextBundle(text, count_tokens(text), [h.record.id for h in chosen_hits], [h.record.id for h in hits], sections)


@dataclass
class Evidence:
    scene_hits: list[Hit]
    memory_hits: list[Hit]
    entity_ids: list[str]
    graph_lines: list[str]


class HybridLongTermMemory:
    def __init__(
        self,
        scene_index: VectorStore,
        memory_index: VectorStore,
        graph: TemporalKnowledgeGraph,
        embedder: Embedder,
        config: RetrievalConfig,
    ):
        self.scene_index = scene_index
        self.memory_index = memory_index
        self.graph = graph
        self.embedder = embedder
        self.config = config

    def _name(self, entity_id: str) -> str:
        entity = self.graph.entities.get(entity_id)
        return entity.name if entity else entity_id

    def _render_fact(self, fact: Fact, cutoff: int) -> str:
        start = self.graph.episode_label(fact.valid_from)
        if fact.valid_to is not None and fact.valid_to <= cutoff:
            interval = f"from {start} until {self.graph.episode_label(fact.valid_to)} (no longer true)"
        else:
            interval = f"since {start}"
        detail = f": {fact.description}" if fact.description else ""
        return f"- {self._name(fact.subject)} {fact.predicate.replace('_', ' ')} {self._name(fact.object)}, {interval}{detail}"

    def _render_event(self, event: Event) -> str:
        names = ", ".join(self._name(p) for p in event.participants)
        who = f" [{names}]" if names else ""
        return f"- {self.graph.episode_label(event.episode_index)} scene {event.scene_index}: {event.description}{who}"

    def _seed_entities(self, query: str, cutoff: int, scene_hits: Sequence[Hit]) -> list[str]:
        entity_ids = self.graph.match_entities(query, cutoff)
        if not entity_ids:
            for hit in scene_hits[:SPEAKER_SEED_HITS]:
                for speaker in hit.record.meta.get("speakers", []):
                    resolved = self.graph.resolve(speaker)
                    if resolved and resolved not in entity_ids:
                        entity_ids.append(resolved)
        return entity_ids[: self.config.max_graph_entities]

    def _graph_lines(self, query: str, cutoff: int, entity_ids: list[str]) -> list[str]:
        if not entity_ids:
            return []
        words = content_words(query) | {w for e in entity_ids for w in normalize_name(self._name(e)).split()}
        seeds = set(entity_ids)
        horizon = max(cutoff, 1)
        scored: list[tuple[float, tuple[int, int, int], str]] = []
        for fact in self.graph.facts_touching(entity_ids, cutoff):
            both = {fact.subject, fact.object} <= seeds and len(seeds) > 1
            text = self._render_fact(fact, cutoff)
            score = 2.0 * both + lexical_overlap(words, text) + 0.2 * fact.valid_from / horizon
            scored.append((score, (0, fact.valid_from, 0), text))
        for event in self.graph.events_for(entity_ids, cutoff):
            shared = len(seeds & set(event.participants))
            text = self._render_event(event)
            score = 1.0 * (shared > 1) + lexical_overlap(words, text) + 0.1 * event.episode_index / horizon
            scored.append((score, (1, event.episode_index, event.scene_index), text))
        ranked = sorted(scored, key=lambda item: -item[0])
        keep = select_within([text for _, _, text in ranked], self.config.graph_tokens)
        return [text for _, _, text in sorted((ranked[i] for i in keep), key=lambda item: item[1])]

    def gather(self, query: str, cutoff: int) -> Evidence:
        vector = self.embedder.embed_query(query)
        scene_hits = self.scene_index.search(vector, self.config.top_k, cutoff)
        memory_hits = self.memory_index.search(vector, self.config.memory_top_k, cutoff)
        entity_ids = self._seed_entities(query, cutoff, scene_hits)
        return Evidence(scene_hits, memory_hits, entity_ids, self._graph_lines(query, cutoff, entity_ids))

    def compose(self, evidence: Evidence, extra_hits: Iterable[Hit] = ()) -> ContextBundle:
        memories = sorted(evidence.memory_hits, key=lambda h: h.record.order_key)
        memory_lines = [f"- {h.record.episode_id} scene {h.record.scene_index}: {h.record.text}" for h in memories]
        return compose_context(
            list(extra_hits) + evidence.scene_hits,
            self.config.context_tokens,
            graph_lines=evidence.graph_lines,
            memory_lines=memory_lines,
            graph_budget=self.config.graph_tokens,
            memory_budget=self.config.memory_tokens,
        )

    def search_scenes(self, query: str, cutoff: int, k: int, exclude_ids: Iterable[str] = ()) -> list[Hit]:
        return self.scene_index.search(self.embedder.embed_query(query), k, cutoff, exclude_ids)

    def forget_entity(self, name: str) -> dict[str, int]:
        entity_id = self.graph.resolve(name)
        if entity_id is None:
            return {"facts": 0, "events": 0, "memories": 0}
        entity = self.graph.entities[entity_id]
        names = {normalize_name(n) for n in [entity.name, *entity.aliases]}
        removed = self.graph.delete_entity(entity_id)
        memories = self.memory_index.delete(lambda record: any(f" {n} " in f" {normalize_name(record.text)} " for n in names))
        return {**removed, "memories": memories}
