# Initial implementation verification

Date: 2026-11-20. Development host: macOS, Apple Silicon. No CUDA device was available.

| Check | Observed result |
| --- | --- |
| Main environment | Python 3.12.12, NumPy 2.5.5, PyTorch 2.15.0 |
| Full tests, including CPU PyTorch | 201 passed; 2 platform-dependent skips |
| Coverage | 92.31% |
| Strict mypy / Ruff lint / formatting | Passed |
| Minimum-dependency installed-wheel environment | Python 3.11.14, NumPy 1.26.4, threadpoolctl 3.2.0; PyTorch absent |
| Tests against that installed wheel | 184 passed; 10 optional/platform skips |
| Binary example | Exact target found, MSE 0 |
| Numeric parameter-fitting example | Fitted [1.80320344, 0.65072078] for synthetic [1.8, 0.65]; final MSE approximately 1.13e-6 |
| Island example | Four populations of 50, two CPU workers, ring migration; best MSE approximately 9.83e-5 |
| PyTorch example | Passed on CPU; independent batched predictions |
| Distribution build | Wheel and source distribution built with uv/Hatchling |

Independent review identified and verified fixes for real-valued objective validation, bfloat16 conversion, representable large losses, exact numeric bounds, CUDA model ownership, and deadlines covering the full IPC request/reply exchange. Regressions include blocked writes, partial replies, child failure, cancellation and completed-result preservation.

The remaining timeout boundary is explicit: caller-side model/configuration serialization and OS process creation cannot be interrupted by the per-request timeout. Serial in-process callbacks are also outside that timeout. Heavy model construction belongs in the child factory.

CUDA ownership, batching, and typed OOM recovery paths were exercised through a CPU-backed test adapter and genuine PyTorch exception types. This does **not** verify CUDA kernels, actual GPU memory pressure, or performance on a CUDA device. The real CUDA smoke test remains skipped until run on suitable hardware.

No real DKO adapter, checkpoint, biological dataset, or experiment was used. Examples validate library mechanics and are not biological findings. CPU/GPU speedup is workload-dependent; example runtimes are not a comparative benchmark.

GitHub Actions runs CPU tests on Linux/macOS/Windows with Python 3.11–3.13, quality checks with PyTorch, and a minimum-dependency job. Consult the actual workflow runs for remote results; this document records local verification only.

The package has not been uploaded to PyPI or TestPyPI.

## PyPI release preparation — 2026-11-20

- Added an explicit manual publishing workflow with validation-only defaults, release-tag checks, isolated artifact upload, and OIDC authentication.
- Added 14 release tests using real temporary Git repositories. Full local suite: **215 passed, 2 skipped**.
- Ruff lint/format checks and strict mypy passed, including the release guard script.
- `uv build --no-sources` built the wheel and source distribution; `twine check --strict` passed for both.
- Installed the built wheel into a fresh Python 3.12 environment outside the checkout. Verified version `0.1.0`, successful optimization, absence of optional PyTorch, and inclusion of the typing marker and MIT license.
- Independently reviewed the tag provenance and upload-permission boundaries; no blocking findings remained.
- Created GitHub environments `pypi` and `testpypi` and verified that both allow deployment only from branch `main`.

At this checkpoint, the PyPI account's pending publisher is still awaiting user configuration. No index upload has been attempted. The publishing guide records the exact fields and release procedure. These checks verify package preparation, not a public release.
