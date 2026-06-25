#!/usr/bin/env python3
"""
Buffett investment report via OpenAI + local Buffett skill (SKILL.md).

Usage:
    set OPENAI_API_KEY=sk-...
    python buffett_openai_report.py AAPL
    python buffett_openai_report.py D05.SI --output dbs_report.md
    python buffett_openai_report.py MSFT --with-references

Environment:
    OPENAI_API_KEY          Required
    BUFFETT_SKILL_DIR       Optional override for skill folder path

Dependencies:
    pip install openai
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

DEFAULT_MODEL = "gpt5.4-mini"
SKILL_DIRNAME = "buffett"
SKILL_FILENAME = "SKILL.md"
REFERENCES_DIR = "references"

USER_PROMPT_TEMPLATE = "You are Warren Buffett. Write a summary report on {ticker}"

SYSTEM_PREAMBLE = """\
You are an investment analyst embodying Warren Buffett's thinking system.
Follow the Buffett skill instructions below precisely, including the 8-question
quick filter and the Standard Output Format (all sections required).

If live market data is unavailable, use your knowledge and clearly label estimates.
Do not invent precise financial figures — state ranges or "verify in latest filing" when uncertain.

--- BUFFETT SKILL ---

"""


def resolve_skill_dir(explicit: str | None = None) -> Path:
    """Find buffett skill directory on any device (project, cwd, or user home)."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if (path / SKILL_FILENAME).is_file():
            return path
        raise FileNotFoundError(
            f"BUFFETT_SKILL_DIR / --skill-dir does not contain {SKILL_FILENAME}: {path}"
        )

    env_dir = os.environ.get("BUFFETT_SKILL_DIR")
    if env_dir:
        return resolve_skill_dir(env_dir)

    script_root = Path(__file__).resolve().parent
    candidates = [
        script_root / "skills" / SKILL_DIRNAME,
        Path.cwd() / "skills" / SKILL_DIRNAME,
        Path.home() / ".local" / "share" / SKILL_DIRNAME,
    ]

    for candidate in candidates:
        if (candidate / SKILL_FILENAME).is_file():
            return candidate.resolve()

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find {SKILL_FILENAME}. Searched:\n  {searched}\n"
        f"Set BUFFETT_SKILL_DIR or pass --skill-dir to the folder containing {SKILL_FILENAME}."
    )


def load_skill(skill_dir: Path, with_references: bool = False) -> str:
    skill_path = skill_dir / SKILL_FILENAME
    content = skill_path.read_text(encoding="utf-8")

    if not with_references:
        return content

    ref_dir = skill_dir / REFERENCES_DIR
    if not ref_dir.is_dir():
        return content

    parts = [content]
    for ref_file in sorted(ref_dir.glob("*.md")):
        parts.append(f"\n\n---\n## Reference: {ref_file.name}\n\n")
        parts.append(ref_file.read_text(encoding="utf-8"))
    return "".join(parts)


def normalize_ticker(raw: str) -> str:
    t = raw.strip().upper()
    if not t:
        raise ValueError("Ticker cannot be empty")
    if len(t) <= 4 and any(c.isdigit() for c in t) and "." not in t:
        return f"{t}.SI"
    return t


def call_openai(
    client: OpenAI,
    *,
    model: str,
    ticker: str,
    skill_text: str,
    temperature: float,
) -> str:
    system_content = SYSTEM_PREAMBLE + skill_text
    user_content = USER_PROMPT_TEMPLATE.format(ticker=ticker)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
    )
    message = response.choices[0].message
    text = message.content
    if not text:
        raise RuntimeError("OpenAI returned an empty response.")
    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Buffett-style report via OpenAI using the local buffett skill."
    )
    parser.add_argument("ticker", help="Ticker symbol (e.g. AAPL, D05.SI)")
    parser.add_argument(
        "-o",
        "--output",
        help="Write report to file (default: stdout)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--skill-dir",
        help="Path to buffett skill folder (overrides auto-discovery)",
    )
    parser.add_argument(
        "--with-references",
        action="store_true",
        help="Include all references/*.md files in the system prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature (default: 0.4)",
    )
    args = parser.parse_args()

    if OpenAI is None:
        print("Error: openai package required.\n  Install: pip install openai", file=sys.stderr)
        return 1

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY environment variable is not set.\n"
            "  Example: set OPENAI_API_KEY=sk-...   (Windows)\n"
            "           export OPENAI_API_KEY=sk-... (macOS/Linux)",
            file=sys.stderr,
        )
        return 1

    try:
        ticker = normalize_ticker(args.ticker)
        skill_dir = resolve_skill_dir(args.skill_dir)
        skill_text = load_skill(skill_dir, with_references=args.with_references)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)

    try:
        report = call_openai(
            client,
            model=args.model,
            ticker=ticker,
            skill_text=skill_text,
            temperature=args.temperature,
        )
    except Exception as exc:
        print(f"Error calling OpenAI API: {exc}", file=sys.stderr)
        return 1

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(report + "\n", encoding="utf-8")
        print(f"Report written to {out_path}", file=sys.stderr)
        print(f"Skill loaded from: {skill_dir}", file=sys.stderr)
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
