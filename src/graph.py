import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx

NON_NAME = re.compile(r"[^\w\s'\-]")
SPACES = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    return SPACES.sub(" ", NON_NAME.sub(" ", name.lower())).strip()


def normalize_predicate(predicate: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", predicate.lower()).strip("_") or "related_to"


@dataclass
class Entity:
    id: str
    name: str
    type: str = "character"
    aliases: list[str] = field(default_factory=list)
    first_episode: int = 0
    last_episode: int = 0
    mentions: int = 0


@dataclass
class Fact:
    id: str
    subject: str
    predicate: str
    object: str
    description: str
    valid_from: int
    valid_to: int | None = None
    provenance: list[tuple[str, int]] = field(default_factory=list)
    invalidated_by: str | None = None
    supersedes: list[str] = field(default_factory=list)

    def visible_at(self, cutoff: int) -> bool:
        return self.valid_from <= cutoff

    def valid_at(self, cutoff: int) -> bool:
        return self.visible_at(cutoff) and (self.valid_to is None or self.valid_to > cutoff)

    def sources_at(self, cutoff: int) -> list[str]:
        return [scene_id for scene_id, episode in self.provenance if episode <= cutoff]


@dataclass
class Event:
    id: str
    description: str
    episode_index: int
    scene_index: int
    scene_id: str
    participants: list[str] = field(default_factory=list)

    @property
    def order_key(self) -> tuple[int, int]:
        return (self.episode_index, self.scene_index)


class TemporalKnowledgeGraph:
    def __init__(self, episodes: list[str] | None = None):
        self.episodes = list(episodes or [])
        self.entities: dict[str, Entity] = {}
        self.facts: dict[str, Fact] = {}
        self.events: dict[str, Event] = {}
        self._aliases: dict[str, str] = {}
        self._events_by_entity: dict[str, list[str]] = defaultdict(list)
        self._graph = nx.MultiDiGraph()
        self._counter = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter:06d}"

    def episode_label(self, index: int) -> str:
        return self.episodes[index] if 0 <= index < len(self.episodes) else f"ep{index}"

    def resolve(self, name: str) -> str | None:
        return self._aliases.get(normalize_name(name))

    def upsert_entity(self, name: str, episode_index: int, type: str = "character", aliases: Iterable[str] = ()) -> Entity:
        names = [name, *aliases]
        entity_id = next((self.resolve(n) for n in names if self.resolve(n)), None)
        if entity_id is None:
            entity_id = normalize_name(name).replace(" ", "_") or self._next_id("E")
            if entity_id in self.entities:
                entity_id = f"{entity_id}_{self._next_id('')}"
            entity = Entity(entity_id, name.strip(), type, [], episode_index, episode_index)
            self.entities[entity_id] = entity
            self._graph.add_node(entity_id)
        entity = self.entities[entity_id]
        entity.first_episode = min(entity.first_episode, episode_index)
        entity.last_episode = max(entity.last_episode, episode_index)
        entity.mentions += 1
        for alias in names:
            key = normalize_name(alias)
            if key and key not in self._aliases:
                self._aliases[key] = entity_id
                if alias.strip() != entity.name and alias.strip() not in entity.aliases:
                    entity.aliases.append(alias.strip())
        return entity

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        description: str,
        episode_index: int,
        scene_id: str,
        supersedes: Iterable[str] = (),
    ) -> Fact:
        fact = Fact(
            id=self._next_id("F"),
            subject=subject,
            predicate=normalize_predicate(predicate),
            object=object,
            description=description.strip(),
            valid_from=episode_index,
            provenance=[(scene_id, episode_index)],
            supersedes=list(supersedes),
        )
        self.facts[fact.id] = fact
        self._graph.add_edge(subject, object, key=fact.id)
        return fact

    def add_provenance(self, fact_id: str, scene_id: str, episode_index: int) -> None:
        fact = self.facts[fact_id]
        if all(scene_id != existing for existing, _ in fact.provenance):
            fact.provenance.append((scene_id, episode_index))

    def invalidate(self, fact_id: str, episode_index: int, by: str | None = None) -> None:
        fact = self.facts[fact_id]
        if fact.valid_to is None or episode_index < fact.valid_to:
            fact.valid_to = max(episode_index, fact.valid_from)
            fact.invalidated_by = by

    def add_event(
        self, description: str, participants: list[str], episode_index: int, scene_index: int, scene_id: str
    ) -> Event:
        event = Event(self._next_id("V"), description.strip(), episode_index, scene_index, scene_id, list(dict.fromkeys(participants)))
        self.events[event.id] = event
        for participant in event.participants:
            self._events_by_entity[participant].append(event.id)
        return event

    def facts_touching(self, entity_ids: Iterable[str], cutoff: int, only_valid: bool = False) -> list[Fact]:
        wanted = set(entity_ids)
        found: dict[str, Fact] = {}
        for entity_id in wanted:
            if entity_id not in self._graph:
                continue
            for _, _, key in list(self._graph.out_edges(entity_id, keys=True)) + list(self._graph.in_edges(entity_id, keys=True)):
                fact = self.facts.get(key)
                if fact and (fact.valid_at(cutoff) if only_valid else fact.visible_at(cutoff)):
                    found[fact.id] = fact
        return sorted(found.values(), key=lambda fact: (fact.valid_from, fact.id))

    def facts_between(self, first: str, second: str, cutoff: int, only_valid: bool = False) -> list[Fact]:
        return [f for f in self.facts_touching([first], cutoff, only_valid) if {f.subject, f.object} == {first, second}]

    def events_for(self, entity_ids: Iterable[str], cutoff: int) -> list[Event]:
        found = {
            event_id: self.events[event_id]
            for entity_id in set(entity_ids)
            for event_id in self._events_by_entity.get(entity_id, [])
            if event_id in self.events and self.events[event_id].episode_index <= cutoff
        }
        return sorted(found.values(), key=lambda event: event.order_key)

    def top_entity_names(self, limit: int) -> list[str]:
        return [entity.name for entity in sorted(self.entities.values(), key=lambda e: -e.mentions)[:limit]]

    def match_entities(self, text: str, cutoff: int) -> list[str]:
        normalized = f" {normalize_name(text)} "
        hits: list[tuple[int, str]] = []
        taken: list[tuple[int, int]] = []
        for alias in sorted(self._aliases, key=len, reverse=True):
            entity_id = self._aliases[alias]
            entity = self.entities.get(entity_id)
            if entity is None or entity.first_episode > cutoff:
                continue
            for match in re.finditer(rf"(?<=\s){re.escape(alias)}(?=[\s'])", normalized):
                span = (match.start(), match.end())
                if any(span[0] < end and start < span[1] for start, end in taken):
                    continue
                taken.append(span)
                hits.append((span[0], entity_id))
        ordered = [entity_id for _, entity_id in sorted(hits)]
        return list(dict.fromkeys(ordered))

    def delete_entity(self, entity_id: str) -> dict[str, int]:
        if entity_id not in self.entities:
            return {"facts": 0, "events": 0}
        facts = [fact_id for fact_id, fact in self.facts.items() if entity_id in (fact.subject, fact.object)]
        for fact_id in facts:
            del self.facts[fact_id]
        removed_events = 0
        for event_id in self._events_by_entity.pop(entity_id, []):
            event = self.events.get(event_id)
            if event is None:
                continue
            event.participants = [p for p in event.participants if p != entity_id]
            if not event.participants:
                del self.events[event_id]
                removed_events += 1
        self._aliases = {alias: target for alias, target in self._aliases.items() if target != entity_id}
        del self.entities[entity_id]
        if entity_id in self._graph:
            self._graph.remove_node(entity_id)
        return {"facts": len(facts), "events": removed_events}

    def stats(self) -> dict[str, int]:
        return {
            "entities": len(self.entities),
            "facts": len(self.facts),
            "invalidated_facts": sum(1 for fact in self.facts.values() if fact.valid_to is not None),
            "events": len(self.events),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "episodes": self.episodes,
            "counter": self._counter,
            "entities": [asdict(entity) for entity in self.entities.values()],
            "aliases": self._aliases,
            "facts": [asdict(fact) for fact in self.facts.values()],
            "events": [asdict(event) for event in self.events.values()],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, path: Path) -> "TemporalKnowledgeGraph":
        data = json.loads(path.read_text())
        graph = cls(data["episodes"])
        graph._counter = data["counter"]
        graph._aliases = data["aliases"]
        for row in data["entities"]:
            graph.entities[row["id"]] = Entity(**row)
            graph._graph.add_node(row["id"])
        for row in data["facts"]:
            graph.facts[row["id"]] = Fact(**{**row, "provenance": [tuple(p) for p in row["provenance"]]})
            graph._graph.add_edge(row["subject"], row["object"], key=row["id"])
        for row in data["events"]:
            graph.events[row["id"]] = Event(**row)
            for participant in row["participants"]:
                graph._events_by_entity[participant].append(row["id"])
        return graph
