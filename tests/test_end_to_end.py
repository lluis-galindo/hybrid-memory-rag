from src.cli import main
from src.corpus import ingest_directory, save_scenes
from src.memory import build_long_term_memory, build_scene_index
from src.questions import load_questions
from src.utils import read_jsonl
from src.workspace import Workspace

from .conftest import SAMPLE_DIR, SAMPLE_QUESTIONS


def test_full_offline_pipeline(tmp_path, offline_config_path, offline_models):
    def run_cli(*args):
        main(["--config", str(offline_config_path), *args])

    run_cli("ingest")
    run_cli("index")
    run_cli("build-memory")
    questions = str(SAMPLE_QUESTIONS)
    run_cli("run", "--questions", questions, "--run-id", "smoke")
    run_cli("run", "--questions", questions, "--run-id", "smoke")
    run_cli("judge", "--questions", questions, "--run-id", "smoke")
    run_cli("report", "--questions", questions, "--run-id", "smoke")

    run_dir = tmp_path / "runs" / "smoke"
    n = len(load_questions(SAMPLE_QUESTIONS))
    for system in ("B0", "B1", "C1", "C2"):
        rows = list(read_jsonl(run_dir / f"{system}.jsonl"))
        assert len(rows) == n
        assert not any(row["error"] for row in rows)
    report = (run_dir / "report.md").read_text()
    assert all(f"| {system} | {n} |" in report for system in ("B0", "B1", "C1", "C2"))
    assert "## H1" in report and "## H4" in report
    assert len(list(read_jsonl(run_dir / "judgements.jsonl"))) == 4 * n

    c2 = {row["question_id"]: row for row in read_jsonl(run_dir / "C2.jsonl")}
    nodes = [step["node"] for step in c2["hl-cross-01"]["trace"]]
    assert nodes[:3] == ["orchestrator_intake", "memory_retrieve", "critic_route"]
    assert "corpus_search" in nodes and nodes[-1] == "memory_write"


def test_no_leakage_of_future_episodes(offline_config, offline_models):
    workspace = Workspace(offline_config)
    scenes = ingest_directory(SAMPLE_DIR, "harbor_lights", offline_config.segmentation)
    save_scenes(scenes, offline_config.scenes_path)
    build_scene_index(scenes, workspace.embedder).save(offline_config.scene_index_dir)
    result = build_long_term_memory(scenes, workspace.llm, workspace.embedder)
    result.save(offline_config.graph_path, offline_config.memory_index_dir, offline_config.consolidation_log_path)

    cutoff = 2
    for name in ("B0", "B1", "C1", "C2"):
        answer = workspace.system(name).answer("Who took the lighthouse key? Elena Pike Mara", cutoff)
        episodes = {scene_id.split("_sc")[0] for scene_id in answer.context_scene_ids + answer.ranked_scene_ids}
        assert episodes <= {"S01E01", "S01E02", "S01E03"}, name
    evidence = workspace.long_term.gather("Elena Jonah key", cutoff)
    assert all("S01E04" not in line and "S01E05" not in line for line in evidence.graph_lines)
