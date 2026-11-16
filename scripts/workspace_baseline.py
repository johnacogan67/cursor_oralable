#!/usr/bin/env python3
"""Report the Git baseline for the Oralable multi-repository workspace.

This is intentionally read-only. It does not run builds, modify Git state,
change ignore rules, or contact network services.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR.parent / "workspace.manifest.json"


def run_git(repo: Path, *args: str) -> str:
    """Run a Git command in a repository and return its stdout."""
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def match_pattern(path: str, pattern: str) -> bool:
    """Match a repository-relative POSIX path against a manifest pattern."""
    if fnmatch.fnmatch(path, pattern):
        return True
    if pattern.endswith("/**"):
        prefix = pattern[:-3].rstrip("/")
        return path == prefix or path.startswith(prefix + "/")
    return False


def status_entries(repo: Path) -> list[dict[str, str]]:
    """Return porcelain-v1 status entries without changing the worktree."""
    entries: list[dict[str, str]] = []
    for line in run_git(repo, "status", "--porcelain=v1").splitlines():
        if len(line) < 4:
            continue
        code = line[:2]
        path = line[3:]
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[-1]
        entries.append({"code": code, "path": path})
    return entries


def branch_status(repo: Path) -> dict[str, str | None]:
    """Collect local Git identity without fetching from remotes."""
    try:
        branch = run_git(repo, "branch", "--show-current").strip() or None
        head = run_git(repo, "rev-parse", "HEAD").strip()
        short_head = run_git(repo, "rev-parse", "--short", "HEAD").strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Not a readable Git repository: {repo}\n{error.stderr.strip()}") from error

    try:
        upstream = run_git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}").strip()
    except subprocess.CalledProcessError:
        upstream = None

    return {
        "branch": branch,
        "head": head,
        "short_head": short_head,
        "upstream": upstream,
    }


def classify_entry(entry: dict[str, str], patterns: list[str]) -> str:
    """Classify a porcelain entry as generated, untracked source, or tracked change."""
    if entry["code"] != "??":
        return "tracked_change"
    if any(match_pattern(entry["path"], pattern) for pattern in patterns):
        return "generated_untracked"
    return "reviewable_untracked"


def inspect_repository(root: Path, repository: dict[str, Any]) -> dict[str, Any]:
    """Build one structured repository report."""
    repo_path = (root / repository["path"]).resolve()
    if not repo_path.exists():
        return {
            "id": repository["id"],
            "path": str(repo_path),
            "role": repository["role"],
            "present": False,
            "error": "directory is missing",
        }

    identity = branch_status(repo_path)
    patterns = list(repository.get("generated_patterns", []))
    entries = status_entries(repo_path)
    classified = [
        {**entry, "classification": classify_entry(entry, patterns)}
        for entry in entries
    ]
    counts = Counter(entry["classification"] for entry in classified)

    return {
        "id": repository["id"],
        "path": str(repo_path),
        "role": repository["role"],
        "present": True,
        "git": identity,
        "release_inputs": repository.get("release_inputs", []),
        "generated_patterns": patterns,
        "summary": {
            "tracked_changes": counts["tracked_change"],
            "reviewable_untracked": counts["reviewable_untracked"],
            "generated_untracked": counts["generated_untracked"],
            "total_status_entries": len(classified),
        },
        "entries": classified,
    }


def build_report(manifest_path: Path, include_external: bool) -> dict[str, Any]:
    """Load the manifest and inspect each selected local checkout."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported workspace manifest schema_version")

    root = manifest_path.parent.resolve()
    selected_ids = set(manifest["workspace"]["canonical_repository_ids"])
    if include_external:
        selected_ids.update(manifest["workspace"].get("external_repository_ids", []))

    repositories = [
        inspect_repository(root, repo)
        for repo in manifest["repositories"]
        if repo["id"] in selected_ids
    ]
    totals = Counter()
    for repository in repositories:
        for key, value in repository.get("summary", {}).items():
            totals[key] += value

    return {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "workspace": manifest["workspace"],
        "release_baseline_policy": manifest["release_baseline_policy"],
        "contract_owners": manifest["contract_owners"],
        "repositories": repositories,
        "totals": dict(totals),
    }


def markdown_report(report: dict[str, Any]) -> str:
    """Render a deterministic human-readable summary from the JSON report."""
    lines = [
        "# Oralable Workspace Baseline Status",
        "",
        f"Manifest: `{report['manifest']}`",
        "",
        "## Summary",
        "",
        f"- Tracked changes: **{report['totals'].get('tracked_changes', 0)}**",
        f"- Reviewable untracked paths: **{report['totals'].get('reviewable_untracked', 0)}**",
        f"- Generated/local untracked paths: **{report['totals'].get('generated_untracked', 0)}**",
        "",
        "## Repositories",
        "",
    ]

    for repository in report["repositories"]:
        lines.append(f"### {repository['id']}")
        lines.append("")
        lines.append(f"- Path: `{repository['path']}`")
        lines.append(f"- Role: {repository['role']}")
        if not repository["present"]:
            lines.append(f"- Status: **missing** — {repository['error']}")
            lines.append("")
            continue
        git = repository["git"]
        summary = repository["summary"]
        lines.extend([
            f"- Git: `{git['branch'] or 'detached'}` at `{git['short_head']}`",
            f"- Upstream: `{git['upstream'] or 'none'}`",
            f"- Tracked changes: **{summary['tracked_changes']}**",
            f"- Reviewable untracked: **{summary['reviewable_untracked']}**",
            f"- Generated/local untracked: **{summary['generated_untracked']}**",
            "",
        ])
        reviewable = [entry["path"] for entry in repository["entries"] if entry["classification"] == "reviewable_untracked"]
        if reviewable:
            lines.append("Reviewable untracked paths:")
            lines.extend(f"- `{path}`" for path in reviewable)
            lines.append("")

    lines.extend([
        "## Release-baseline interpretation",
        "",
        "A non-zero status does not mean a path is wrong. It means the path must be reviewed, committed, deliberately removed, or explicitly accepted before a release baseline is tagged. Generated paths are separated so that build products and local evaluation output do not obscure source work.",
        "",
        "This checker does not modify Git state, build software, flash hardware, or contact external services.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to workspace.manifest.json (default: %(default)s)",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="markdown",
        help="Output format (default: %(default)s)",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Include nearby website repositories in addition to the four canonical repositories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the report to this path instead of stdout.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 if any tracked or reviewable-untracked paths need review.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args.manifest.resolve(), args.include_external)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"workspace-baseline: {error}", file=sys.stderr)
        return 1

    output = json.dumps(report, indent=2, sort_keys=True) + "\n" if args.format == "json" else markdown_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")

    if args.strict:
        totals = report["totals"]
        if totals.get("tracked_changes", 0) or totals.get("reviewable_untracked", 0):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
