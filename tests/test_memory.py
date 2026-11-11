import json

from src.graph import TemporalKnowledgeGraph
from src.memory import Consolidator, ExtractedRelation, SceneExtraction, ShortTermMemory
from src.vector_store import VectorRecord, VectorStore

from .conftest import make_scene
from .fakes import HashingEmbedder, ScriptedLLM, offline_responder


def test_vector_store_filters_future_episodes(tmp_path):
    embedder = HashingEmbedder(64)
    texts = ["the key is lost", "the key is found", "bread and ovens"]
    records = [VectorRecord(f"r{i}", t, "scene", f"S01E0{i + 1}", i, 0) for i, t in enumerate(texts)]
    store = VectorStore(records, embedder.embed_documents(texts), embedder.name)
    hits = store.search(embedder.embed_query("key"), k=3, max_episode_index=0)
    assert [h.record.id for h in hits] == ["r0"]
    store.save(tmp_path / "idx")
    loaded = VectorStore.load(tmp_path / "idx", embedder.name)
    assert len(loaded) == 3
    assert loaded.search(embedder.embed_query("key"), k=1, exclude_ids=["r0", "r1"])[0].record.id == "r2"


def test_graph_temporal_validity_and_roundtrip(tmp_path):
    graph = TemporalKnowledgeGraph(["S01E01", "S01E02", "S01E03"])
    mara = graph.upsert_entity("Mara Quinn", 0, aliases=["Mara"]).id
    theo = graph.upsert_entity("Theo", 0).id
    fact = graph.add_fact(mara, "dating", theo, "", 1, "S01E02_sc000")
    graph.invalidate(fact.id, 2)
    assert not fact.visible_at(0)
    assert fact.valid_at(1)
    assert not fact.valid_at(2) and fact.visible_at(2)
    assert graph.match_entities("Did Mara and Theo break up?", 2) == [mara, theo]
    graph.save(tmp_path / "g.json")
    loaded = TemporalKnowledgeGraph.load(tmp_path / "g.json")
    assert loaded.facts[fact.id].valid_to == 2
    assert loaded.resolve("mara") == mara


def test_graph_delete_entity_removes_facts_and_aliases():
    graph = TemporalKnowledgeGraph(["S01E01"])
    a = graph.upsert_entity("Ann", 0).id
    b = graph.upsert_entity("Bob", 0).id
    graph.add_fact(a, "friend_of", b, "", 0, "x")
    graph.add_event("Ann waves", [a], 0, 0, "x")
    removed = graph.delete_entity(a)
    assert removed == {"facts": 1, "events": 1}
    assert graph.resolve("Ann") is None and not graph.facts


def test_consolidation_update_invalidates_previous_fact():
    graph = TemporalKnowledgeGraph(["S01E01", "S01E02", "S01E03"])

    def responder(messages, schema):
        fact_id = messages[-1]["content"].split("Existing facts:\n")[1].split(":")[0]
        return json.dumps({"operation": "UPDATE", "target_fact_ids": [fact_id], "reason": "status changed"})

    consolidator = Consolidator(graph, ScriptedLLM(responder))
    first = make_scene(1, 0, [("Mara", "Yes to you, Theo."), ("Theo", "Great.")])
    later = make_scene(2, 0, [("Mara", "We're done, Theo."), ("Theo", "So that's it?")])
    consolidator.apply(first, SceneExtraction(summary="", relations=[ExtractedRelation(subject="Mara", predicate="dating", object="Theo")]))
    consolidator.apply(later, SceneExtraction(summary="", relations=[ExtractedRelation(subject="Mara", predicate="broke up with", object="Theo")]))
    assert consolidator.operations["UPDATE"] == 1
    old, new = sorted(graph.facts.values(), key=lambda f: f.valid_from)
    assert old.valid_to == 2 and old.invalidated_by == new.id
    assert new.predicate == "broke_up_with" and new.supersedes == [old.id]
    assert old.valid_at(1) and not old.valid_at(2)


def test_consolidation_exact_duplicate_is_noop_without_llm():
    graph = TemporalKnowledgeGraph(["S01E01", "S01E02"])
    consolidator = Consolidator(graph, ScriptedLLM(lambda m, s: (_ for _ in ()).throw(AssertionError("no LLM call"))))
    relation = ExtractedRelation(subject="Jonah", predicate="sibling_of", object="Theo")
    consolidator.apply(make_scene(0, 0, []), SceneExtraction(summary="", relations=[relation]))
    consolidator.apply(make_scene(1, 0, []), SceneExtraction(summary="", relations=[relation]))
    assert consolidator.operations == {"ADD": 1, "NOOP": 1}
    (fact,) = graph.facts.values()
    assert fact.sources_at(0) == ["S01E01_sc000"]
    assert fact.sources_at(1) == ["S01E01_sc000", "S01E02_sc000"]


def test_short_term_memory_respects_budget():
    memory = ShortTermMemory(60, 2, ScriptedLLM(offline_responder))
    for i in range(10):
        memory.add("user", f"question number {i} about Mara and the lighthouse key")
    assert memory.tokens <= 60
    assert len(memory.turns) < 10
    assert memory.summary
    memory.clear()
    assert memory.render() == ""
