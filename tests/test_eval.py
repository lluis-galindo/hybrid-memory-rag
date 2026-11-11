from src.evaluation import answer_scores, exact_match, mcnemar_exact, paired_bootstrap, recall_at_k, token_f1
from src.llm import extract_json
from src.memory import build_scene_index
from src.questions import Question, QuestionGenerator, answer_adds_information, distance_bucket, episode_distance, in_story, stratified_sample
from src.retrieval import compose_context
from src.systems import is_abstention
from src.vector_store import Hit, VectorRecord

from .fakes import HashingEmbedder, ScriptedLLM, offline_responder


def test_answer_metrics():
    assert exact_match("The Crust & Crumb!", ["Crust & Crumb"]) == 1.0
    assert 0 < token_f1("Mara Quinn the baker", ["Mara Quinn"]) < 1
    assert answer_scores("I don't know", True, ["x"], unanswerable=True) == {"em": 1.0, "f1": 1.0}
    assert answer_scores("Mara", True, ["Mara"], unanswerable=False) == {"em": 0.0, "f1": 0.0}
    assert recall_at_k(["a", "b", "c"], ["c", "d"], 2) == 0.0
    assert recall_at_k(["a", "b", "c"], ["c", "d"], 3) == 0.5
    assert recall_at_k(["a"], [], 3) is None


def test_abstention_detection():
    assert is_abstention("I don't know.")
    assert is_abstention("The context does not say")
    assert not is_abstention("Mara knows")


def test_distance_buckets():
    index = {f"S01E{i:02d}": i - 1 for i in range(1, 11)}
    q = Question("q", "cross_episode", "?", "a", "S01E09", ["S01E02_sc000", "S01E08_sc001"])
    assert episode_distance(q, index) == 7
    assert distance_bucket(7) == ">5" and distance_bucket(0) == "same" and distance_bucket(3) == "1-5"


def test_stratified_sample_covers_types():
    questions = [Question(f"{t}-{i}", t, "?", "a", "S01E01") for t in ("fact", "order", "unanswerable") for i in range(10)]
    sample = stratified_sample(questions, 6)
    assert len(sample) == 6
    assert {q.type for q in sample} == {"fact", "order", "unanswerable"}


def test_paired_statistics():
    a = [1.0] * 30 + [0.0] * 10
    b = [0.0] * 30 + [0.0] * 10
    difference, low, _ = paired_bootstrap(a, b, resamples=2000)
    assert difference == 0.75 and low > 0
    assert mcnemar_exact([x == 1 for x in a], [x == 1 for x in b]) < 1e-6
    assert mcnemar_exact([True, False], [True, False]) == 1.0


def test_extract_json_handles_think_and_fences():
    assert extract_json('<think>hmm {"x": 0}</think>```json\n{"a": 1}\n```') == '{"a": 1}'


def test_compose_context_respects_budget_and_orders_chronologically():
    hits = [
        Hit(VectorRecord(f"S01E0{e}_sc000", "word " * 200, "scene", f"S01E0{e}", e - 1, 0), 1.0 - e / 10)
        for e in (5, 2, 9, 1)
    ]
    bundle = compose_context(hits, budget=700, graph_lines=["- fact"], graph_budget=100)
    assert bundle.tokens <= 700
    assert bundle.context_scene_ids == ["S01E02_sc000", "S01E05_sc000"]
    assert bundle.ranked_scene_ids[0] == "S01E05_sc000"
    assert bundle.sections["graph"] == 1


def test_recap_generator_links_evidence_scenes(sample_scenes):
    embedder = HashingEmbedder(64)
    recaps = {scene.episode_id: "Mara opens the bakery and Theo arrives in town to visit Jonah." for scene in sample_scenes}
    generator = QuestionGenerator(ScriptedLLM(offline_responder), sample_scenes, "harbor_lights", recaps, build_scene_index(sample_scenes, embedder), embedder, facts_per_recap=2)
    questions = generator.generate(2, ("fact", "cross_episode", "order"))
    assert questions
    by_episode = {scene.scene_id: scene.episode_id for scene in sample_scenes}
    for question in questions:
        assert question.source.startswith("generated:recap_")
        assert question.evidence_scene_ids and all(scene_id in by_episode for scene_id in question.evidence_scene_ids)
        if question.type != "fact":
            first, second = (by_episode[s] for s in question.evidence_scene_ids)
            assert first < second


def test_recap_questions_about_production_are_rejected():
    assert not in_story("Who is portrayed by guest star Faye Dunaway?", "Ariana Kane")
    assert not in_story("How did it change between the earlier and later recap?", "x")
    assert in_story("Who does Sydney meet in Taipei?", "Noah Hicks")


def test_generated_answers_must_add_information():
    echo = Question("a", "fact", "What event causes Sydney to begin losing trust in her father?", "Sydney's trust in her father begins to fade.", "S01E01")
    good = Question("b", "fact", "Who tells Jack the truth about Emily's death?", "Sloane", "S01E01")
    assert not answer_adds_information(echo)
    assert answer_adds_information(good)
