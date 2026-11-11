import json

from src.config import SegmentationConfig
from src.corpus import SceneBreak, Utterance, ingest_directory, load_episode, parse_episode_number, parse_transcript_lines, prepare_show, segment_blocks


def test_episode_numbers():
    assert parse_episode_number("Friends.S02E07") == (2, 7)
    assert parse_episode_number("03x12 - The One") == (3, 12)
    assert parse_episode_number("notes.txt") is None


def test_transcript_markers_directions_and_continuations():
    blocks = parse_transcript_lines(
        ["[Scene: Central Perk]", "Monica: Hi.", "and welcome", "(Ross enters.)", "INT. APARTMENT - NIGHT", "Dr. Green: Rachel!"]
    )
    assert isinstance(blocks[0], SceneBreak) and blocks[0].heading == "Central Perk"
    assert blocks[1] == Utterance(text="Hi. and welcome", speaker="Monica")
    assert blocks[2].speaker is None
    assert isinstance(blocks[3], SceneBreak)
    assert blocks[4].speaker == "Dr. Green"


def test_long_scenes_are_split_and_short_ones_merged():
    blocks = [SceneBreak("a")] + [Utterance(text="word " * 40, speaker="A") for _ in range(10)] + [SceneBreak("b"), Utterance(text="ok", speaker="B")]
    scenes = segment_blocks(blocks, SegmentationConfig(min_scene_tokens=20, max_scene_tokens=200))
    assert len(scenes) >= 2
    assert all(sum(len(u.render()) // 4 for u in s[1]) <= 220 for s in scenes)
    assert scenes[-1][1][-1].text == "ok"


def test_summscreen_json(tmp_path):
    path = tmp_path / "S02E05.json"
    path.write_text(json.dumps({"Episode Title": "Pilot", "Transcript": ["[Scene: Bar]", "Sam: Hi."]}))
    blocks, title = load_episode(path)
    assert title == "Pilot" and blocks == [SceneBreak("Bar"), Utterance("Hi.", "Sam")]


def test_ingest_sample(sample_scenes):
    assert {s.episode_id for s in sample_scenes} == {f"S01E0{i}" for i in range(1, 7)}
    assert [s.episode_index for s in sample_scenes] == sorted(s.episode_index for s in sample_scenes)
    assert sample_scenes[0].scene_id == "S01E01_sc000"
    assert "Mara" in sample_scenes[0].speakers


def test_ingest_rejects_duplicates(tmp_path):
    (tmp_path / "S01E01.txt").write_text("A: hi")
    (tmp_path / "1x01.txt").write_text("A: hi")
    try:
        ingest_directory(tmp_path, "x", SegmentationConfig())
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("expected duplicate error")


def test_location_directions_open_scenes_when_enabled():
    lines = ["SYDNEY: Hi.", "(At the C.I.A. office, Vaughn waits.)", "VAUGHN: Late again.", "(He sighs.)", "[SCENE_BREAK]", "JACK: Sydney."]
    plain = parse_transcript_lines(lines)
    assert sum(isinstance(b, SceneBreak) for b in plain) == 1
    assert next(b for b in plain if isinstance(b, SceneBreak)).heading is None
    blocks = parse_transcript_lines(lines, location_breaks=True)
    breaks = [b for b in blocks if isinstance(b, SceneBreak)]
    assert [b.heading for b in breaks] == ["At the C.I.A. office", None]
    assert blocks[2].text.startswith("(At the C.I.A.")
    assert blocks[4].text == "(He sighs.)"


def test_prepare_summscreen_show(tmp_path):
    source = tmp_path / "fd"
    source.mkdir()
    for name in ("Spy_01x01.json", "Spy_01x03.json", "Spy_02x01.json", "Spy_Kids_01x01.json"):
        (source / name).write_text(json.dumps({"Episode Title": name, "Transcript": ["A: hi there"], "Recap": ["Something happens."]}))
    manifest = prepare_show(source, "Spy", {1}, tmp_path / "raw" / "spy")
    assert sorted(p.name for p in (tmp_path / "raw" / "spy").iterdir()) == ["S01E01.json", "S01E03.json"]
    assert manifest["missing"] == ["S01E02"]
    assert (tmp_path / "raw" / "spy_recaps.jsonl").exists()
