from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .corpus import Scene, episode_ids
from .embeddings import Embedder
from .graph import Fact, TemporalKnowledgeGraph, normalize_predicate
from .llm import LLM, StructuredOutputError
from .utils import count_tokens, truncate_to_tokens, write_jsonl
from .vector_store import VectorRecord, VectorStore

EXTRACTION_SYSTEM = """You build a knowledge graph of a TV series, one scene at a time.
Return a JSON object with:
- summary: one or two sentences about what happens in the scene.
- entities: characters, places and organizations that matter in the scene. Use full names when known and reuse the names of known characters.
- relations: lasting facts between two entities that hold after this scene, such as dating, married_to, friend_of, sibling_of, parent_of, works_at, lives_in, enemy_of, broke_up_with. predicate is short snake_case.
- events: the important things that happen in the scene, each with the characters involved.
Only include what the scene states or clearly implies. Do not use outside knowledge about the series."""

CONSOLIDATION_SYSTEM = """You maintain the long-term memory of a TV series. A new fact was extracted from the latest scene.
Compare it with the facts currently stored about the same characters and choose one operation:
- ADD: the new fact is new information and the existing facts are still true.
- UPDATE: the new fact replaces existing facts that are no longer true (for example, a relationship status changed). List them in target_fact_ids.
- INVALIDATE: the new fact only says that existing facts stopped being true. List them in target_fact_ids.
- NOOP: the new fact is already captured by an existing fact. Put that fact in target_fact_ids.
Return a JSON object."""

SUMMARY_SYSTEM = """Summarise this conversation about a TV series in at most five short sentences.
Keep the characters, episodes and facts that were discussed, and any open question of the user."""

MAX_EXTRACTION_TOKENS = 1500
MAX_CANDIDATES = 12
KNOWN_ENTITY_HINTS = 30


class ExtractedEntity(BaseModel):
    name: str
    type: Literal["character", "place", "organization", "object", "other"] = "character"
    aliases: list[str] = Field(default_factory=list)


class ExtractedRelation(BaseModel):
    subject: str
    predicate: str
    object: str
    description: str = ""


class ExtractedEvent(BaseModel):
    description: str
    participants: list[str] = Field(default_factory=list)


class SceneExtraction(BaseModel):
    summary: str
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    events: list[ExtractedEvent] = Field(default_factory=list)


class ConsolidationDecision(BaseModel):
    operation: Literal["ADD", "UPDATE", "INVALIDATE", "NOOP"]
    target_fact_ids: list[str] = Field(default_factory=list)
    reason: str = ""


def scene_header(scene: Scene) -> str:
    title = f" ({scene.episode_title})" if scene.episode_title else ""
    heading = f" - {scene.heading}" if scene.heading else ""
    return f"Series: {scene.series}\nEpisode: {scene.episode_id}{title}\nScene {scene.scene_index}{heading}"


def extract_scene(llm: LLM, scene: Scene, known_entities: list[str] | None = None) -> SceneExtraction:
    known = f"\nKnown characters so far: {', '.join(known_entities)}" if known_entities else ""
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": f"{scene_header(scene)}{known}\n\n{scene.text}"},
    ]
    return llm.structured(messages, SceneExtraction, role="extraction", max_tokens=MAX_EXTRACTION_TOKENS)


class Consolidator:
    def __init__(self, graph: TemporalKnowledgeGraph, llm: LLM):
        self.graph = graph
        self.llm = llm
        self.operations: Counter = Counter()
        self.log: list[dict[str, Any]] = []

    def _render(self, fact: Fact) -> str:
        entities = self.graph.entities
        subject = entities[fact.subject].name if fact.subject in entities else fact.subject
        obj = entities[fact.object].name if fact.object in entities else fact.object
        detail = f" - {fact.description}" if fact.description else ""
        return f"{fact.id}: {subject} {fact.predicate} {obj} (since {self.graph.episode_label(fact.valid_from)}){detail}"

    def _candidates(self, subject: str, predicate: str, obj: str, episode_index: int) -> list[Fact]:
        between = self.graph.facts_between(subject, obj, episode_index, only_valid=True)
        same_predicate = [
            fact
            for fact in self.graph.facts_touching([subject], episode_index, only_valid=True)
            if fact.predicate == predicate and fact.subject == subject
        ]
        merged = {fact.id: fact for fact in between + same_predicate}
        return sorted(merged.values(), key=lambda fact: -fact.valid_from)[:MAX_CANDIDATES]

    def _decide(self, scene: Scene, new: str, candidates: list[Fact]) -> ConsolidationDecision:
        existing = "\n".join(self._render(fact) for fact in candidates)
        messages = [
            {"role": "system", "content": CONSOLIDATION_SYSTEM},
            {"role": "user", "content": f"Existing facts:\n{existing}\n\nNew fact (from {scene.episode_id}): {new}"},
        ]
        try:
            return self.llm.structured(messages, ConsolidationDecision, role="consolidation")
        except StructuredOutputError as error:
            return ConsolidationDecision(operation="ADD", reason=f"fallback: {error}")

    def apply(self, scene: Scene, extraction: SceneExtraction) -> None:
        episode = scene.episode_index
        for entity in extraction.entities:
            if entity.name.strip():
                self.graph.upsert_entity(entity.name, episode, entity.type, entity.aliases)
        for speaker in scene.speakers:
            self.graph.upsert_entity(speaker, episode)
        for event in extraction.events:
            participants = [self.graph.upsert_entity(name, episode).id for name in event.participants if name.strip()]
            self.graph.add_event(event.description, participants, episode, scene.scene_index, scene.scene_id)
        for relation in extraction.relations:
            if relation.subject.strip() and relation.object.strip():
                subject = self.graph.upsert_entity(relation.subject, episode).id
                obj = self.graph.upsert_entity(relation.object, episode).id
                self._consolidate(scene, subject, normalize_predicate(relation.predicate), obj, relation.description)

    def _consolidate(self, scene: Scene, subject: str, predicate: str, obj: str, description: str) -> None:
        episode = scene.episode_index
        candidates = self._candidates(subject, predicate, obj, episode)
        exact = next((f for f in candidates if (f.subject, f.predicate, f.object) == (subject, predicate, obj)), None)
        new_text = f"{self.graph.entities[subject].name} {predicate} {self.graph.entities[obj].name}"
        if description:
            new_text += f" - {description}"
        if exact is not None:
            decision = ConsolidationDecision(operation="NOOP", target_fact_ids=[exact.id], reason="exact match")
        elif not candidates:
            decision = ConsolidationDecision(operation="ADD", reason="no related facts")
        else:
            decision = self._decide(scene, new_text, candidates)
        allowed = {fact.id for fact in candidates}
        targets = [fact_id for fact_id in decision.target_fact_ids if fact_id in allowed]
        operation = decision.operation if targets else "ADD"
        created = None
        if operation == "ADD":
            created = self.graph.add_fact(subject, predicate, obj, description, episode, scene.scene_id)
        elif operation == "UPDATE":
            created = self.graph.add_fact(subject, predicate, obj, description, episode, scene.scene_id, supersedes=targets)
            for fact_id in targets:
                self.graph.invalidate(fact_id, episode, by=created.id)
        elif operation == "INVALIDATE":
            for fact_id in targets:
                self.graph.invalidate(fact_id, episode, by=scene.scene_id)
        else:
            for fact_id in targets:
                self.graph.add_provenance(fact_id, scene.scene_id, episode)
        self.operations[operation] += 1
        self.log.append(
            {
                "scene_id": scene.scene_id,
                "operation": operation,
                "proposed": decision.operation,
                "new_fact": new_text,
                "created": created.id if created else None,
                "targets": targets,
                "candidates": [fact.id for fact in candidates],
                "reason": decision.reason,
            }
        )


def _store_from(records: list[VectorRecord], documents: list[str], embedder: Embedder) -> VectorStore:
    return VectorStore(records, embedder.embed_documents(documents), embedder.name)


def build_scene_index(scenes: list[Scene], embedder: Embedder) -> VectorStore:
    records = [
        VectorRecord(s.scene_id, s.text, "scene", s.episode_id, s.episode_index, s.scene_index, {"speakers": s.speakers, "heading": s.heading})
        for s in scenes
    ]
    documents = [f"{s.episode_id}{f' {s.heading}' if s.heading else ''}\n{s.text}" for s in scenes]
    return _store_from(records, documents, embedder)


@dataclass
class MemoryBuildResult:
    graph: TemporalKnowledgeGraph
    memory_index: VectorStore
    consolidator: Consolidator
    failures: list[str] = field(default_factory=list)

    def save(self, graph_path: Path, memory_dir: Path, log_path: Path) -> None:
        self.graph.save(graph_path)
        self.memory_index.save(memory_dir)
        write_jsonl(log_path, self.consolidator.log)


def build_long_term_memory(scenes: list[Scene], llm: LLM, embedder: Embedder) -> MemoryBuildResult:
    ordered = sorted(scenes, key=lambda scene: scene.order_key)
    graph = TemporalKnowledgeGraph(episode_ids(ordered))
    consolidator = Consolidator(graph, llm)
    records: list[VectorRecord] = []
    failures: list[str] = []
    for position, scene in enumerate(ordered):
        if position % 10 == 0:
            print(f"[memory] {position}/{len(ordered)} {scene.scene_id}", flush=True)
        try:
            extraction = extract_scene(llm, scene, graph.top_entity_names(KNOWN_ENTITY_HINTS))
        except StructuredOutputError:
            failures.append(scene.scene_id)
            continue
        consolidator.apply(scene, extraction)
        if extraction.summary.strip():
            records.append(
                VectorRecord(
                    f"{scene.scene_id}#memory",
                    extraction.summary.strip(),
                    "memory",
                    scene.episode_id,
                    scene.episode_index,
                    scene.scene_index,
                    {"scene_id": scene.scene_id, "speakers": scene.speakers},
                )
            )
    return MemoryBuildResult(graph, _store_from(records, [r.text for r in records], embedder), consolidator, failures)


@dataclass
class Turn:
    role: str
    content: str

    def render(self) -> str:
        return f"{self.role.capitalize()}: {self.content}"


class ShortTermMemory:
    def __init__(self, budget_tokens: int, keep_recent_turns: int, summarizer: LLM):
        self.budget_tokens = budget_tokens
        self.keep_recent_turns = keep_recent_turns
        self.summarizer = summarizer
        self.summary = ""
        self.turns: list[Turn] = []

    @property
    def tokens(self) -> int:
        return count_tokens(self.render())

    def add(self, role: str, content: str) -> None:
        self.turns.append(Turn(role, content))
        self._compress()

    def clear(self) -> None:
        self.summary = ""
        self.turns = []

    def _summarise(self, turns: list[Turn]) -> str:
        dialogue = "\n".join(turn.render() for turn in turns)
        prior = f"Earlier summary: {self.summary}\n\n" if self.summary else ""
        messages = [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": f"{prior}{dialogue}"}]
        return truncate_to_tokens(self.summarizer.complete(messages, role="short_term").text.strip(), self.budget_tokens // 3)

    def _compress(self) -> None:
        while self.tokens > self.budget_tokens and len(self.turns) > self.keep_recent_turns:
            overflow = len(self.turns) - self.keep_recent_turns
            evicted, self.turns = self.turns[:overflow], self.turns[overflow:]
            self.summary = self._summarise(evicted)
        while self.tokens > self.budget_tokens and len(self.turns) > 1:
            self.turns = self.turns[1:]

    def render(self) -> str:
        summary = [f"Summary of the earlier conversation: {self.summary}"] if self.summary else []
        return "\n".join(summary + [turn.render() for turn in self.turns])
