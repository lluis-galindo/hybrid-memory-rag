import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import SegmentationConfig
from .utils import count_tokens, read_jsonl, write_jsonl

EPISODE_NUMBER = re.compile(r"[Ss](\d{1,2})[ ._-]?[Ee](\d{1,3})|(?<!\d)(\d{1,2})x(\d{1,3})(?!\d)")
MARKUP = re.compile(r"<[^>]+>|\{[^}]*\}")
SCENE_MARKER = re.compile(
    r"^\s*(?:\[\s*scene[_ ]?break\s*\]|\[\s*scene\b[^\]]*\]|\(\s*scene\b[^)]*\)|scene\s*:.*|(?:INT|EXT)\.\s.*|[-=*~]{3,})\s*$",
    re.IGNORECASE,
)
MARKER_PREFIX = re.compile(r"^\s*[\[(]?\s*scene\s*:?\s*", re.IGNORECASE)
DIRECTION = re.compile(r"^\s*[\[(].*[\])]\s*$")
SPEAKER_LINE = re.compile(r"^\s*([A-Z][\w.'\-]*(?:\s+[A-Z&][\w.'\-]*){0,3})\s*(?:\([^)]*\))?\s*:\s*(.+)$")
LOCATION_CUE = re.compile(
    r"^\s*[\[(]\s*(?:(?:back\s+)?(?:in|at|inside|outside|on)\s|back\s+(?:to|in|at)\b|cut\s+to\b|flashback\b|present\b|meanwhile\b|elsewhere\b)",
    re.IGNORECASE,
)
SUMMSCREEN_FILE = re.compile(r"^(?P<show>.+)_(?P<season>\d+)x(?P<episode>\d+)\.json$")
SUMMSCREEN_SOURCE = "SummScreen (Chen et al., ACL 2022), ForeverDreaming subset, untokenized release"


def make_episode_id(season: int, episode: int) -> str:
    return f"S{season:02d}E{episode:02d}"


def episode_of_scene(scene_id: str) -> str:
    return scene_id.split("_sc")[0]


@dataclass
class Utterance:
    text: str
    speaker: str | None = None

    def render(self) -> str:
        return f"{self.speaker}: {self.text}" if self.speaker else self.text


@dataclass
class SceneBreak:
    heading: str | None = None


Block = Utterance | SceneBreak
RawScene = tuple[str | None, list[Utterance]]


@dataclass
class Scene:
    scene_id: str
    series: str
    episode_id: str
    season: int
    episode: int
    episode_index: int
    scene_index: int
    utterances: list[Utterance] = field(default_factory=list)
    heading: str | None = None
    episode_title: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(utterance.render() for utterance in self.utterances)

    @property
    def speakers(self) -> list[str]:
        return sorted({u.speaker for u in self.utterances if u.speaker})

    @property
    def order_key(self) -> tuple[int, int]:
        return (self.episode_index, self.scene_index)


def save_scenes(scenes: list[Scene], path: Path) -> None:
    write_jsonl(path, (asdict(scene) for scene in scenes))


def load_scenes(path: Path) -> list[Scene]:
    scenes = [Scene(**{**row, "utterances": [Utterance(**u) for u in row["utterances"]]}) for row in read_jsonl(path)]
    return sorted(scenes, key=lambda scene: scene.order_key)


def episode_ids(scenes: list[Scene]) -> list[str]:
    ordered = {scene.episode_index: scene.episode_id for scene in scenes}
    return [ordered[index] for index in sorted(ordered)]


def parse_episode_number(text: str) -> tuple[int, int] | None:
    match = EPISODE_NUMBER.search(text)
    if not match:
        return None
    season, episode = (int(group) for group in match.groups() if group)
    return season, episode


def _heading_of(marker: str) -> str | None:
    text = marker.strip()
    if re.fullmatch(r"[-=*~]{3,}", text) or re.fullmatch(r"[\[(]\s*scene[_ ]?break\s*[\])]", text, re.IGNORECASE):
        return None
    return MARKER_PREFIX.sub("", text).strip().strip("[]()").strip() or None


def _location_of(direction: str) -> str:
    return re.split(r"[,;:)\]]|\.(?=\s+[A-Z]|\s*$)", direction.strip().lstrip("[(").strip(), maxsplit=1)[0].strip()[:80]


def parse_transcript_lines(lines: list[str], location_breaks: bool = False) -> list[Block]:
    blocks: list[Block] = []
    for raw in lines:
        line = re.sub(r"\s+", " ", MARKUP.sub("", raw)).strip()
        if not line:
            continue
        if SCENE_MARKER.match(line):
            blocks.append(SceneBreak(_heading_of(line)))
        elif DIRECTION.match(line):
            if location_breaks and LOCATION_CUE.match(line):
                blocks.append(SceneBreak(_location_of(line) or None))
            blocks.append(Utterance(text=line))
        elif match := SPEAKER_LINE.match(line):
            blocks.append(Utterance(text=match.group(2).strip(), speaker=match.group(1).strip()))
        elif blocks and isinstance(blocks[-1], Utterance) and blocks[-1].speaker:
            blocks[-1].text = f"{blocks[-1].text} {line}"
        else:
            blocks.append(Utterance(text=line))
    return blocks


def load_episode(path: Path, location_breaks: bool = False) -> tuple[list[Block], str | None]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        data = json.loads(text)
        return parse_transcript_lines(data["Transcript"], location_breaks), data.get("Episode Title")
    return parse_transcript_lines(text.split("\n"), location_breaks), None


def _tokens(utterances: list[Utterance]) -> int:
    return sum(count_tokens(u.render()) + 1 for u in utterances)


def _split_long(scene: RawScene, max_tokens: int) -> list[RawScene]:
    heading, utterances = scene
    chunks: list[RawScene] = []
    current: list[Utterance] = []
    for utterance in utterances:
        if current and _tokens(current + [utterance]) > max_tokens:
            chunks.append((heading, current))
            current = []
        current.append(utterance)
    if current:
        chunks.append((heading, current))
    return chunks


def _merge_short(scenes: list[RawScene], config: SegmentationConfig) -> list[RawScene]:
    merged: list[RawScene] = []
    for heading, utterances in scenes:
        size = _tokens(utterances)
        if merged and size < config.min_scene_tokens and _tokens(merged[-1][1]) + size <= config.max_scene_tokens:
            merged[-1] = (merged[-1][0], merged[-1][1] + utterances)
        else:
            merged.append((heading, list(utterances)))
    return merged


def segment_blocks(blocks: list[Block], config: SegmentationConfig) -> list[RawScene]:
    scenes: list[RawScene] = []
    current: list[Utterance] = []
    heading: str | None = None
    for block in blocks:
        if isinstance(block, SceneBreak):
            if current:
                scenes.append((heading, current))
                current = []
            heading = block.heading
        else:
            current.append(block)
    if current:
        scenes.append((heading, current))
    split = [chunk for scene in scenes for chunk in _split_long(scene, config.max_scene_tokens)]
    return _merge_short(split, config)


def ingest_directory(input_dir: Path, series: str, config: SegmentationConfig) -> list[Scene]:
    numbered = {}
    for path in sorted(input_dir.iterdir()):
        if path.suffix in (".txt", ".json"):
            number = parse_episode_number(path.stem)
            if number is None:
                raise ValueError(f"Cannot determine season/episode for {path.name}; name it like S01E03.txt")
            if number in numbered:
                raise ValueError(f"Duplicate episode {number} in {input_dir}")
            numbered[number] = path
    scenes: list[Scene] = []
    for episode_index, (season, episode) in enumerate(sorted(numbered)):
        blocks, title = load_episode(numbered[season, episode], config.location_breaks)
        episode_id = make_episode_id(season, episode)
        for scene_index, (heading, utterances) in enumerate(segment_blocks(blocks, config)):
            scene_id = f"{episode_id}_sc{scene_index:03d}"
            scenes.append(Scene(scene_id, series, episode_id, season, episode, episode_index, scene_index, utterances, heading, title))
    return scenes


def corpus_stats(scenes: list[Scene]) -> dict[str, float]:
    tokens = [count_tokens(scene.text) for scene in scenes]
    return {
        "episodes": len({scene.episode_id for scene in scenes}),
        "scenes": len(scenes),
        "utterances": sum(len(scene.utterances) for scene in scenes),
        "tokens": sum(tokens),
        "scenes_with_speakers": sum(1 for scene in scenes if scene.speakers) / len(scenes),
        "mean_scene_tokens": sum(tokens) / len(tokens),
    }


def find_show_episodes(source_dir: Path, show: str, seasons: set[int]) -> list[tuple[int, int, Path]]:
    found = []
    for path in source_dir.glob(f"{show}_*.json"):
        match = SUMMSCREEN_FILE.match(path.name)
        if match and match.group("show") == show and int(match.group("season")) in seasons:
            found.append((int(match.group("season")), int(match.group("episode")), path))
    return sorted(found)


def missing_episodes(episodes: list[tuple[int, int, Path]]) -> list[str]:
    present = {(season, episode) for season, episode, _ in episodes}
    last = {season: max(e for s, e in present if s == season) for season, _ in present}
    return [make_episode_id(s, e) for s in sorted(last) for e in range(1, last[s] + 1) if (s, e) not in present]


def prepare_show(source_dir: Path, show: str, seasons: set[int], out_dir: Path) -> dict[str, Any]:
    episodes = find_show_episodes(source_dir, show, seasons)
    if not episodes:
        raise ValueError(f"No episodes of {show} for seasons {sorted(seasons)} in {source_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    recaps = []
    for season, episode, path in episodes:
        raw = path.read_bytes()
        data = json.loads(raw)
        episode_id = make_episode_id(season, episode)
        transcript = data.get("Transcript") or []
        recap = data.get("Recap") or []
        title = data.get("Episode Title") or ""
        (out_dir / f"{episode_id}.json").write_bytes(raw)
        records.append(
            {
                "episode_id": episode_id,
                "title": title,
                "source_file": path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "transcript_author": data.get("Transcript Author") or "",
                "transcript_lines": len(transcript),
                "transcript_words": sum(len(line.split()) for line in transcript),
                "recap_words": sum(len(line.split()) for line in recap),
            }
        )
        recaps.append({"episode_id": episode_id, "title": title, "recap": " ".join(recap).strip()})
    write_jsonl(out_dir.parent / f"{out_dir.name}_recaps.jsonl", recaps)
    manifest = {
        "source": SUMMSCREEN_SOURCE,
        "show": show,
        "seasons": sorted(seasons),
        "episodes": len(records),
        "missing": missing_episodes(episodes),
        "transcript_words": sum(r["transcript_words"] for r in records),
        "files": records,
    }
    (out_dir.parent / f"{out_dir.name}_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest
