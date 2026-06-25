import os
import re
from pathlib import Path

from openai import OpenAI

SKILL_FOLDER_NAME = "finance-sentiment-skill"
SKILL_MANIFEST_NAME = "finance-sentiment-skill.md"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
HOSTED_SKILL_ID = os.environ.get("FINANCE_SENTIMENT_SKILL_ID")
SKILL_PATH_ENV = "FINANCE_SENTIMENT_SKILL_PATH"
SKILLS_DIR_ENV = "SKILLS_DIR"

# The client automatically picks up the OPENAI_API_KEY environment variable.
client = OpenAI()


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _skill_search_paths() -> list[Path]:
    paths: list[Path] = []

    if env_path := os.environ.get(SKILL_PATH_ENV):
        paths.append(Path(env_path).expanduser())

    if skills_dir := os.environ.get(SKILLS_DIR_ENV):
        paths.append(Path(skills_dir).expanduser() / SKILL_FOLDER_NAME)

    paths.extend(
        [
            Path.home() / "skills" / SKILL_FOLDER_NAME,
            _module_dir() / "skills" / SKILL_FOLDER_NAME,
        ]
    )

    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)
    return unique_paths


def find_skill_manifest(skill_dir: Path) -> Path:
    matches = sorted(
        entry
        for entry in skill_dir.iterdir()
        if entry.is_file() and entry.name.lower() == SKILL_MANIFEST_NAME.lower()
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple {SKILL_MANIFEST_NAME} files found in {skill_dir}")

    raise FileNotFoundError(
        f"No {SKILL_MANIFEST_NAME} found in {skill_dir}. "
        f"Add the skill manifest as {SKILL_MANIFEST_NAME}."
    )


def resolve_skill_dir() -> Path:
    checked: list[str] = []
    for candidate in _skill_search_paths():
        checked.append(str(candidate))
        if not candidate.is_dir():
            continue
        try:
            find_skill_manifest(candidate)
        except (FileNotFoundError, ValueError):
            continue
        return candidate

    checked_list = "\n  - ".join(checked)
    raise FileNotFoundError(
        f"Could not find a valid {SKILL_FOLDER_NAME} folder with {SKILL_MANIFEST_NAME}. Checked:\n"
        f"  - {checked_list}\n"
        f"Set {SKILL_PATH_ENV} to the skill folder path, {SKILLS_DIR_ENV} to a "
        f"shared skills library, or install the skill to ~/skills/{SKILL_FOLDER_NAME}."
    )


def _read_manifest_text(skill_dir: Path) -> str:
    return find_skill_manifest(skill_dir).read_text(encoding="utf-8")


def read_skill_name(skill_dir: Path) -> str:
    match = re.search(r"^name:\s*(.+)$", _read_manifest_text(skill_dir), re.MULTILINE)
    if match:
        return match.group(1).strip().strip("\"'")
    return SKILL_FOLDER_NAME


def read_skill_description(skill_dir: Path) -> str:
    text = _read_manifest_text(skill_dir)
    match = re.search(
        r"^description:\s*(?:>\s*\n((?:[ \t].+\n?)+)|(.+))$",
        text,
        re.MULTILINE,
    )
    if match:
        if match.group(1):
            return " ".join(line.strip() for line in match.group(1).splitlines())
        return match.group(2).strip().strip("\"'")

    return (
        "Analyze market sentiment for a stock ticker using recent news, "
        "price action, and market context."
    )


def verify_skill_setup() -> dict[str, str]:
    skill_dir = resolve_skill_dir()
    manifest = find_skill_manifest(skill_dir)
    return {
        "skill_name": read_skill_name(skill_dir),
        "skill_dir": str(skill_dir),
        "manifest": str(manifest),
        "description": read_skill_description(skill_dir),
        "mode": "hosted" if HOSTED_SKILL_ID else "local",
    }


def _shell_tools():
    if HOSTED_SKILL_ID:
        return [
            {
                "type": "shell",
                "environment": {
                    "type": "container_auto",
                    "skills": [
                        {
                            "type": "skill_reference",
                            "skill_id": HOSTED_SKILL_ID,
                            "version": "latest",
                        }
                    ],
                },
            }
        ]

    skill_dir = resolve_skill_dir()
    skill_name = read_skill_name(skill_dir)
    return [
        {
            "type": "shell",
            "environment": {
                "type": "local",
                "skills": [
                    {
                        "name": skill_name,
                        "description": read_skill_description(skill_dir),
                        "path": str(skill_dir),
                    }
                ],
            },
        }
    ]


def fill_ticker_sentiment(ticker: str) -> str:
    skill_dir = resolve_skill_dir()
    skill_name = read_skill_name(skill_dir)
    verify_skill_setup()
    response = client.responses.create(
        model=DEFAULT_MODEL,
        tools=_shell_tools(),
        input=f"Use the {skill_name} skill to produce a sentiment report on {ticker}.",
    )
    return response.output_text


if __name__ == "__main__":
    import sys

    if "--verify-skill" in sys.argv:
        setup = verify_skill_setup()
        print("Skill setup OK")
        for key, value in setup.items():
            print(f"{key}: {value}")
        raise SystemExit(0)

    ticker = input("Ticker: ")
    print(fill_ticker_sentiment(ticker))
