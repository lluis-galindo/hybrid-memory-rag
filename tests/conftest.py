from pathlib import Path

import pytest
import yaml

from src.config import Config, SegmentationConfig
from src.corpus import Scene, Utterance, ingest_directory
from src.workspace import Workspace

from .fakes import HashingEmbedder, ScriptedLLM, offline_responder

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample" / "harbor_lights"
SAMPLE_QUESTIONS = ROOT / "data" / "sample" / "harbor_lights_questions.jsonl"


@pytest.fixture
def offline_settings(tmp_path):
    return {
        "series": "harbor_lights",
        "llm": {"model": "offline"},
        "judge": {"model": "offline"},
        "embedding": {"model": "offline"},
        "paths": {
            "raw_dir": str(SAMPLE_DIR.parent),
            "processed_dir": str(tmp_path / "processed"),
            "runs_dir": str(tmp_path / "runs"),
            "cache_dir": str(tmp_path / "cache"),
        },
    }


@pytest.fixture
def offline_config(offline_settings):
    return Config.model_validate(offline_settings)


@pytest.fixture
def offline_config_path(tmp_path, offline_settings):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(offline_settings))
    return path


@pytest.fixture
def offline_models(monkeypatch):
    monkeypatch.setattr(Workspace, "_build_llm", lambda self, config: ScriptedLLM(offline_responder))
    monkeypatch.setattr(Workspace, "embedder", HashingEmbedder())


@pytest.fixture
def sample_scenes():
    return ingest_directory(SAMPLE_DIR, "harbor_lights", SegmentationConfig())


def make_scene(episode_index: int, scene_index: int, lines: list[tuple[str, str]]) -> Scene:
    episode_id = f"S01E{episode_index + 1:02d}"
    return Scene(
        scene_id=f"{episode_id}_sc{scene_index:03d}",
        series="test",
        episode_id=episode_id,
        season=1,
        episode=episode_index + 1,
        episode_index=episode_index,
        scene_index=scene_index,
        utterances=[Utterance(text=text, speaker=speaker) for speaker, text in lines],
    )
