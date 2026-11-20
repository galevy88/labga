"""Validate source versions and the Git provenance of a proposed release.

This command only reads files and Git history. Publishing targets additionally
require a manual workflow run from main and an existing version tag on main.
"""

import argparse
import ast
import subprocess
import tomllib
from pathlib import Path


def source_version(root: Path) -> str:
    """Read version metadata without importing or executing the package."""
    with (root / "pyproject.toml").open("rb") as metadata_file:
        metadata = tomllib.load(metadata_file)
    version = metadata.get("project", {}).get("version")
    if not isinstance(version, str) or not version or version.strip() != version:
        raise ValueError("pyproject.toml must declare a nonempty project.version string")

    tree = ast.parse((root / "src" / "labga" / "__init__.py").read_text(encoding="utf-8"))
    package_versions: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                package_versions.append(node.value.value)
    if package_versions != [version]:
        raise ValueError("labga.__version__ must be a single literal matching project.version")
    return version


def git(root: Path, *arguments: str) -> str:
    """Run a read-only Git query and report failures as validation errors."""
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"Git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def validate_release(
    root: Path,
    *,
    target: str,
    release_tag: str,
    event_name: str,
    workflow_ref: str,
) -> str:
    """Require matching versions and, when supplied, an exact release tag on main."""
    version = source_version(root)
    if target != "validate":
        if event_name != "workflow_dispatch" or workflow_ref != "refs/heads/main":
            raise ValueError("Publishing requires workflow_dispatch from refs/heads/main")
        if not release_tag:
            raise ValueError("Publishing requires an explicit release tag")
    if release_tag:
        expected_tag = f"v{version}"
        if release_tag != expected_tag:
            raise ValueError(f"Release tag must be {expected_tag}, matching project.version")
        tag_commit = git(root, "rev-parse", "--verify", f"refs/tags/{release_tag}^{{commit}}")
        checkout_commit = git(root, "rev-parse", "--verify", "HEAD^{commit}")
        if tag_commit != checkout_commit:
            raise ValueError("Release tag must identify the checked out commit (checkout HEAD)")
        git(root, "merge-base", "--is-ancestor", tag_commit, "refs/remotes/origin/main")
    return version


def main() -> None:
    """Parse workflow inputs and fail the job when release validation fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--target", choices=("validate", "testpypi", "pypi"), default="validate")
    parser.add_argument("--release-tag", default="")
    parser.add_argument("--event-name", default="")
    parser.add_argument("--workflow-ref", default="")
    args = parser.parse_args()
    try:
        version = validate_release(
            args.root,
            target=args.target,
            release_tag=args.release_tag,
            event_name=args.event_name,
            workflow_ref=args.workflow_ref,
        )
    except (OSError, SyntaxError, ValueError) as error:
        parser.exit(1, f"Release validation failed: {error}\n")
    print(f"Validated labga {version} for {args.target}")


if __name__ == "__main__":
    main()
