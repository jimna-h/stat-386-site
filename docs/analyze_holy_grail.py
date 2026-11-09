"""Extract aggregate dialogue features from a locally obtained screenplay PDF.

The script does not download the screenplay or save dialogue text. Its parser
uses conservative speaker-label heuristics; inspect the PDF and extracted
counts before drawing conclusions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pymupdf


INLINE_CUE = re.compile(
    r"^\s*([A-Z][A-Z0-9 .&'’()\-]{0,38})\s*:\s*(\S.*)$"
)
SCENE_HEADING = re.compile(
    r"^(?:INT/EXT\.|INT\.|EXT\.|I/E\.)(?:\s|$)", re.IGNORECASE
)
NON_CHARACTER_CUES = {
    "CONTINUED",
    "CONT'D",
    "CUT TO",
    "FADE IN",
    "FADE OUT",
    "THE END",
    "TITLE",
}
MODIFIER = re.compile(r"\s*\((?:CONT['’]?D|CONTINUED|V\.?O\.?|O\.?S\.?)\)\s*$", re.I)
WORD = re.compile(r"\b[\w]+(?:['’][\w]+)?\b", re.UNICODE)


def is_character_cue(line: str, next_line: str | None) -> str | None:
    """Return a likely uppercase character cue, not ordinary uppercase text."""
    candidate = line.strip()
    if not candidate or len(candidate) > 40 or next_line is None:
        return None
    if candidate.isdigit() or SCENE_HEADING.match(candidate):
        return None
    candidate = MODIFIER.sub("", candidate).strip()
    if candidate.upper() in NON_CHARACTER_CUES:
        return None
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9 .&'’()\-]*", candidate):
        return None
    letters = [char for char in candidate if char.isalpha()]
    if not letters or not any(char.isalpha() for char in next_line):
        return None
    if next_line.strip().isupper():
        return None
    return candidate


def extract_turns(pdf_path: Path) -> tuple[pd.DataFrame, int, int]:
    """Extract turns and page/scene metadata without retaining dialogue text."""
    records: list[dict[str, Any]] = []
    scene_number = 0
    scene_headings_found = 0
    active_scene: int | None = None
    active_speaker: str | None = None
    active_page = 1
    active_text: list[str] = []

    def finish_turn() -> None:
        nonlocal active_speaker, active_text
        if active_speaker and active_text:
            speech = " ".join(active_text)
            words = WORD.findall(speech)
            records.append(
                {
                    "turn_number": len(records) + 1,
                    "page_number": active_page,
                    "scene_number": active_scene,
                    "speaker": active_speaker,
                    "word_count": len(words),
                    "character_count": len(speech),
                    "question_count": speech.count("?"),
                    "exclamation_count": speech.count("!"),
                }
            )
        active_speaker = None
        active_text = []

    try:
        document = pymupdf.open(pdf_path)
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ValueError(f"Could not open PDF '{pdf_path}': {exc}") from exc

    if document.is_encrypted:
        document.close()
        raise ValueError("The PDF is encrypted; provide an accessible local copy.")

    page_count = len(document)
    with document:
        for page_number, page in enumerate(document, start=1):
            lines = [line.strip() for line in page.get_text("text", sort=True).splitlines()]
            lines = [line for line in lines if line]
            for index, line in enumerate(lines):
                if line.isdigit():
                    continue
                if SCENE_HEADING.match(line):
                    finish_turn()
                    scene_number += 1
                    scene_headings_found += 1
                    active_scene = scene_number
                    continue

                inline_match = INLINE_CUE.match(line)
                if inline_match:
                    finish_turn()
                    active_speaker = MODIFIER.sub("", inline_match.group(1)).strip().title()
                    active_page = page_number
                    active_text = [inline_match.group(2).strip()]
                    continue

                next_line = lines[index + 1] if index + 1 < len(lines) else None
                cue = is_character_cue(line, next_line)
                if cue:
                    finish_turn()
                    active_speaker = MODIFIER.sub("", cue).strip().title()
                    active_page = page_number
                    continue

                if active_speaker:
                    active_text.append(line)

    finish_turn()
    turns = pd.DataFrame.from_records(records)
    if not turns.empty:
        turns["scene_number"] = turns["scene_number"].astype("Int64")
    return turns, page_count, scene_headings_found


def make_plots(turns: pd.DataFrame, speaker_summary: pd.DataFrame, out_dir: Path) -> None:
    """Save plots made only from turn-level counts and labels."""
    plt.figure(figsize=(8, 5))
    plt.hist(turns["word_count"], bins=25, color="#4472C4", edgecolor="white")
    plt.xlabel("Words per dialogue turn")
    plt.ylabel("Number of turns")
    plt.title("Dialogue-turn length distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "turn_length_distribution.png", dpi=160)
    plt.close()

    selected = speaker_summary.head(10).sort_values("median_words_per_turn")
    if not selected.empty:
        plt.figure(figsize=(9, max(4, 0.42 * len(selected))))
        plt.barh(selected["speaker"], selected["median_words_per_turn"], color="#70AD47")
        plt.xlabel("Median words per turn")
        plt.ylabel("Speaker (top 10 by number of turns)")
        plt.title("Typical dialogue-turn length by speaker")
        plt.tight_layout()
        plt.savefig(out_dir / "speaker_turn_lengths.png", dpi=160)
        plt.close()

    page_summary = turns.groupby("page_number", as_index=False)["word_count"].mean()
    plt.figure(figsize=(9, 5))
    plt.plot(page_summary["page_number"], page_summary["word_count"], color="#ED7D31")
    plt.xlabel("PDF page number")
    plt.ylabel("Mean words per dialogue turn")
    plt.title("Dialogue-turn length across PDF pages")
    plt.tight_layout()
    plt.savefig(out_dir / "turn_length_by_page.png", dpi=160)
    plt.close()


def analyze(pdf_path: Path, out_dir: Path, min_speaker_turns: int) -> dict[str, Any]:
    turns, page_count, scene_headings_found = extract_turns(pdf_path)
    if turns.empty:
        raise ValueError(
            "No dialogue turns were identified. The PDF may use a different "
            "layout; inspect its text extraction and adjust the cue rules."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    turns.to_csv(out_dir / "dialogue_turn_features.csv", index=False)
    speaker_summary = (
        turns.groupby("speaker", as_index=False)
        .agg(
            turn_count=("turn_number", "size"),
            total_words=("word_count", "sum"),
            mean_words_per_turn=("word_count", "mean"),
            median_words_per_turn=("word_count", "median"),
            mean_questions_per_turn=("question_count", "mean"),
            mean_exclamations_per_turn=("exclamation_count", "mean"),
        )
        .sort_values(["turn_count", "speaker"], ascending=[False, True])
    )
    speaker_summary.to_csv(out_dir / "speaker_summary.csv", index=False)
    make_plots(turns, speaker_summary, out_dir)

    summary: dict[str, Any] = {
        "source_file": pdf_path.name,
        "pdf_pages": page_count,
        "dialogue_turns": int(len(turns)),
        "distinct_speakers": int(turns["speaker"].nunique()),
        "scene_headings_detected": scene_headings_found,
        "turns_with_scene_number": int(turns["scene_number"].notna().sum()),
        "mean_words_per_turn": round(float(turns["word_count"].mean()), 2),
        "median_words_per_turn": float(turns["word_count"].median()),
        "maximum_words_per_turn": int(turns["word_count"].max()),
        "speakers_with_minimum_turns": int(
            (speaker_summary["turn_count"] >= min_speaker_turns).sum()
        ),
        "minimum_turns_for_speaker_comparison": min_speaker_turns,
        "outputs": [
            "dialogue_turn_features.csv",
            "speaker_summary.csv",
            "turn_length_distribution.png",
            "speaker_turn_lengths.png",
            "turn_length_by_page.png",
        ],
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Analyzed {page_count} PDF pages and identified {len(turns)} dialogue turns.")
    print(f"Recognized {turns['speaker'].nunique()} distinct speaker labels.")
    print(
        f"Mean turn length: {summary['mean_words_per_turn']} words; "
        f"median: {summary['median_words_per_turn']}."
    )
    if scene_headings_found:
        print(f"Recognized {scene_headings_found} scene headings.")
    else:
        print("No standard scene headings were recognized; use PDF pages as a rough location proxy.")
    if len(turns) < 200:
        print(
            "Warning: fewer than 200 turns were identified. Verify the extraction "
            "before using these results."
        )
    print(f"Aggregate results saved to: {out_dir.resolve()}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze dialogue counts in a local screenplay PDF. Obtain and use "
            "the PDF only in accordance with its terms; this tool does not "
            "download or save dialogue text."
        )
    )
    parser.add_argument("pdf", type=Path, help="path to a locally available PDF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("holy-grail-analysis"),
        help="directory for aggregate CSV, JSON, and PNG outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--min-speaker-turns",
        type=int,
        default=5,
        help="minimum turns per speaker to treat comparisons as informative",
    )
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")
    if args.min_speaker_turns < 1:
        parser.error("--min-speaker-turns must be at least 1")
    try:
        analyze(args.pdf, args.output_dir, args.min_speaker_turns)
    except (OSError, ValueError) as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
