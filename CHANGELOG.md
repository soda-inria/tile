# Changelog

The notable changes of each version of TILE. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions
[Semantic Versioning](https://semver.org/spec/v2.0.0.html): before 1.0.0, a
minor version (0.x.0) may change the API.

## [Unreleased]

## [0.2.1] - 2026-11-27

### Fixed

- `TILEFeatures` now standardizes and reduces the cell embeddings in float64,
  whatever their dtype. With float32 cells, the components of a column with
  few distinct cells that should be zero came out as rounding noise that
  differed between training and new rows, which hurt learners that
  standardize their input. Float16 embeddings give very slightly different
  features than before.

## [0.2.0] - 2026-11-26

### Added

- LLMs, each cut at the layer of the probing of `scripts/add_llm.py`:
  Gemma-3-1B, Gemma-3-4B, Gemma-3-12B, Mistral-7B-v0.3, Granite-3.3-8B-Base
  and Nemotron-3-Embed-8B.
- Support for the LLMs that also read images (Gemma-3 from 4B):
  only their LLM is truncated, saved and loaded.
- Learners: Causilo 1.5 (`causilo-1.5`), Seldon (`seldon`), Seldon-mini
  (`seldon-mini`) and InPriora (`inpriora`), with their feature windows and
  the extras `causilo15`, `seldon` and `inpriora`.
- `tile.__version__`.

### Changed

- The learner `causilo` is renamed `causilo-1.0`. `causilo` still works, with
  a `FutureWarning`, and will be removed in a later version.
- A column of cell embeddings that would overflow float16 (as with Gemma) is
  divided by a power of two before it is saved. TILE standardizes each column
  before its PCA, so this changes nothing downstream, and nothing for the LLMs
  whose embeddings fit in float16.
- The layer probing holds the embeddings of every layer in float32 rather than
  float16: twice the memory (about 42 GB for Llama-3.1-8B on the largest
  table). `pool_cells(..., layers=...)` returns float32.
- LLMs whose attention softcaps its scores (Gemma-2) run with the eager
  attention, which applies the softcapping that SDPA leaves out.
- The downloads skip the `consolidated` checkpoints of the Mistral
  repositories.

## [0.1.0] - 2026-10-30

The code released with the paper
([arXiv:2610.17065](https://arxiv.org/abs/2610.17065)).

[Unreleased]: https://github.com/soda-inria/tile/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/soda-inria/tile/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/soda-inria/tile/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/soda-inria/tile/releases/tag/v0.1.0
