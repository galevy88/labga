"""Exercise release validation against real, isolated Git histories."""

import subprocess
import sys
from pathlib import Path

import pytest

RELEASE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_release.py"


def git(repo: Path, *arguments: str) -> str:
    """Run a Git command in a disposable test repository."""
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


@pytest.fixture
def release_repo(tmp_path: Path) -> Path:
    """Create matching source versions and a local snapshot of origin/main."""
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "Release tests")
    git(tmp_path, "config", "user.email", "release-tests@example.invalid")
    git(tmp_path, "config", "commit.gpgsign", "false")
    git(tmp_path, "config", "tag.gpgsign", "false")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n')
    package = tmp_path / "src" / "labga"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.1.0"\n')
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "Initial release")
    git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    return tmp_path


def check_release(
    repo: Path,
    *,
    target: str = "pypi",
    tag: str = "v0.1.0",
    event: str = "workflow_dispatch",
    ref: str = "refs/heads/main",
) -> subprocess.CompletedProcess[str]:
    """Invoke the same CLI used by GitHub Actions without importing the package."""
    return subprocess.run(
        [
            sys.executable,
            str(RELEASE_SCRIPT),
            "--root",
            str(repo),
            "--target",
            target,
            "--release-tag",
            tag,
            "--event-name",
            event,
            "--workflow-ref",
            ref,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize("target", ["testpypi", "pypi"])
def test_publish_accepts_matching_tag_on_main(release_repo: Path, annotated: bool, target: str):
    if annotated:
        git(release_repo, "tag", "-a", "v0.1.0", "-m", "Release 0.1.0")
    else:
        git(release_repo, "tag", "v0.1.0")

    result = check_release(release_repo, target=target)

    assert result.returncode == 0, result.stderr
    assert "0.1.0" in result.stdout


def test_validate_without_a_tag_checks_current_source(release_repo: Path):
    result = check_release(release_repo, target="validate", tag="", event="push")

    assert result.returncode == 0, result.stderr


def test_publish_rejects_tag_for_another_version(release_repo: Path):
    git(release_repo, "tag", "v0.2.0")

    result = check_release(release_repo, tag="v0.2.0")

    assert result.returncode != 0
    assert "v0.1.0" in result.stderr


def test_publish_rejects_tagged_commit_outside_main(release_repo: Path):
    git(release_repo, "switch", "-c", "unmerged")
    (release_repo / "extra.txt").write_text("Unmerged work\n")
    git(release_repo, "add", ".")
    git(release_repo, "commit", "-m", "Unmerged commit")
    git(release_repo, "tag", "v0.1.0")

    result = check_release(release_repo)

    assert result.returncode != 0
    assert "origin/main" in result.stderr


@pytest.mark.parametrize("tag", ["", "v0.1.0"])
def test_publish_rejects_absent_release_tag(release_repo: Path, tag: str):
    result = check_release(release_repo, tag=tag)

    assert result.returncode != 0
    assert "tag" in result.stderr.lower()


def test_release_rejects_source_version_mismatch(release_repo: Path):
    (release_repo / "src" / "labga" / "__init__.py").write_text('__version__ = "0.2.0"\n')

    result = check_release(release_repo, target="validate", tag="")

    assert result.returncode != 0
    assert "__version__" in result.stderr


def test_publish_rejects_tag_different_from_checked_out_commit(release_repo: Path):
    git(release_repo, "tag", "v0.1.0")
    (release_repo / "extra.txt").write_text("Later commit\n")
    git(release_repo, "add", ".")
    git(release_repo, "commit", "-m", "Later main commit")
    git(release_repo, "update-ref", "refs/remotes/origin/main", "HEAD")

    result = check_release(release_repo)

    assert result.returncode != 0
    assert "checkout" in result.stderr.lower()


@pytest.mark.parametrize(
    ("event", "ref"),
    [("push", "refs/heads/main"), ("workflow_dispatch", "refs/heads/feature")],
)
def test_publish_requires_manual_workflow_from_main(release_repo: Path, event: str, ref: str):
    git(release_repo, "tag", "v0.1.0")

    result = check_release(release_repo, event=event, ref=ref)

    assert result.returncode != 0
    assert "workflow_dispatch" in result.stderr


def test_tagged_release_can_be_older_than_main(release_repo: Path):
    git(release_repo, "tag", "v0.1.0")
    release_commit = git(release_repo, "rev-parse", "HEAD")
    (release_repo / "extra.txt").write_text("Later main commit\n")
    git(release_repo, "add", ".")
    git(release_repo, "commit", "-m", "Later main commit")
    git(release_repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    git(release_repo, "checkout", "--detach", release_commit)

    result = check_release(release_repo)

    assert result.returncode == 0, result.stderr
