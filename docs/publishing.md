# Publishing labga

This guide prepares the first `0.1.0` release and subsequent releases. A successful validation run does **not** publish a package. Publication requires an explicit manual workflow dispatch and a configured PyPI trusted publisher.

## One-time account setup

1. Sign in at [PyPI](https://pypi.org/), verify your email, and enable [two-factor authentication](https://docs.pypi.org/your-account/two-factor-authentication/). Keep recovery codes privately.
2. In your account's Publishing settings, add a **pending publisher** for a new project, using the GitHub provider:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `labga` |
   | GitHub owner | `galevy88` |
   | Repository | `labga` |
   | Workflow filename | `publish.yml` |
   | Environment | `pypi` |

3. The project is created by the first successful upload. A pending publisher does not reserve its name. If PyPI rejects the name, resolve that before making a release.

No PyPI password or API token belongs in GitHub secrets, source files, or chat. The workflow uses short-lived OpenID Connect credentials through [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

For an optional rehearsal on **TestPyPI**, create a separate account and pending publisher there. Use the same owner, repository, and workflow, but the environment must be `testpypi`. TestPyPI registration and project ownership are independent of PyPI.

## Repository setup

- Create GitHub environments named `pypi` and `testpypi`.
- In each environment, restrict deployment branches to the branch `main`.
- Keep `.github/workflows/publish.yml` on `main`.
- The workflow has read-only repository permissions by default. Only the isolated publishing job receives `id-token: write`; it downloads checked artifacts and uploads them without checking out or running package code.
- Pull requests, pushes to `main`, and the default manual `validate` mode run checks without publishing.

Do not remove environment restrictions or tag validation to work around a failed release.

## Prepare and validate a version

1. Update `[project].version` in `pyproject.toml` and `labga.__version__` in `src/labga/__init__.py` together. Update this changelog and run `uv lock`.
2. Before the first real release, replace the README's pending-publication installation note and the changelog's candidate label with release wording. Build only after that metadata is final.
3. Run the project checks:

   ```bash
   uv sync --locked --extra torch
   uv run --no-sync pytest
   uv run --no-sync ruff check .
   uv run --no-sync ruff format --check .
   uv run --no-sync mypy
   uv run --no-sync mypy --strict scripts/check_release.py
   uv build --no-sources
   uv run --no-sync twine check --strict dist/*
   ```

4. Review the wheel and source distribution, and install the wheel into a fresh environment outside the checkout. The publishing workflow performs this smoke check automatically as well.
5. Push the changes and require the ordinary CI matrix and publishing validation job to pass before tagging the reviewed commit on `main`.

The release guard checks the package versions, tag spelling, tagged commit, and membership in `main`. The workflow verifies tag ancestry before executing any tagged Python code.

## Publish a reviewed tag

For version `0.1.0`, create and push tag `v0.1.0` on the reviewed commit. Do not move an existing release tag.

In GitHub Actions, open the publishing workflow and choose **Run workflow**:

| Input | Value |
| --- | --- |
| Branch | `main` |
| Target | `pypi` (or `testpypi` for a rehearsal) |
| Release tag | `v0.1.0` |

The workflow builds from that tag, runs tests and distribution checks, installs the wheel in a clean environment, then passes the artifacts to a separate publishing job. Selecting `validate` stops before uploading.

TestPyPI is optional. Its packages and versions do not appear on the production PyPI index. If a rehearsal reveals a package change, make a new version and tag; do not reuse uploaded filenames.

## Verify the public release

After the upload succeeds:

1. Confirm the PyPI project page lists the intended version, author, license, dependencies, README, wheel, and source archive.
2. Install from the production index in a clean environment, without a Git checkout or editable install:

   ```bash
   uv venv /tmp/labga-pypi-check
   uv pip install --python /tmp/labga-pypi-check/bin/python \
     --index-url https://pypi.org/simple "labga==0.1.0"
   cd /tmp
   /tmp/labga-pypi-check/bin/python -c "import labga; print(labga.__version__)"
   ```

3. Run a minimal optimization from that installed package and verify its output. Only then announce that `pip install labga` is available.
4. Record the workflow result and verification in `docs/verification.md`.

## Common failures

- **Invalid publisher:** compare owner, repository, workflow filename, and environment with the table above. Use `publish.yml`, not its full path.
- **Email or 2FA requirement:** complete it in your PyPI account.
- **Tag or ancestry check fails:** tag the reviewed commit on `main`; do not bypass the check.
- **Version or file already exists:** verify whether the earlier upload succeeded. PyPI does not allow reusing a filename after deletion; a changed package needs a new version.
- **Name unavailable:** a pending publisher is not a reservation. Do not upload under a different project name without updating and reviewing package metadata.

References: [PyPI publisher configuration](https://docs.pypi.org/trusted-publishers/adding-a-publisher/), [using a publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/), and [uv package publishing](https://docs.astral.sh/uv/guides/package/).
