# Changelog

## 0.1.0 — first release candidate

Prepared for publication; this heading does not confirm a PyPI upload.

- A single `optimize` API for a standard genetic algorithm and island populations with ring migration.
- Binary and bounded numeric genomes, configurable initialization, per-coordinate bounds, and initial populations.
- `Objective` groups a fixed predictor, target, preprocessing, and MSE, MAE, or a custom loss.
- CPU multiprocessing with one shared worker budget and spawn-compatible predictor factories.
- Optional PyTorch inference with one GPU owner, automatic batching, and typed CUDA out-of-memory fallback.
- Best solutions, predictions, generation history, migration markers, and execution metadata.
- Typed public API, validation, multiprocessing deadlines, and synthetic examples.

This release supports a single computer. Actual CUDA hardware and a real DKO model have not been validated in the initial development environment. See the README for supported interfaces and limitations.
