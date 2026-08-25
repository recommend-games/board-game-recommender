# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `train()` in the `dnn` module: a plain training loop for
  `CollaborativeFilteringModel`, minimising mean squared error with `Adam`
  over shuffled minibatches. No L2 or ranking regularisation yet, and no
  `.npz` export; both are left for a follow-up.

## [4.2.0] - 2026-08-25

### Added

- `dnn` module with a PyTorch implementation of the linear collaborative
  filtering model Turi Create produced, behind a new optional `torch` extra.
  Model only for now: no training loop, no `.npz` export.
- Tests for the PyTorch `CollaborativeFilteringModel`, including one pinning
  down that its scores match what `LightGamesRecommender` serves. They skip
  where the optional `torch` extra is not installed.

### Changed

- Require Python 3.12 or newer, dropping 3.9 through 3.11. Every core
  dependency had already moved past 3.9, so this unlocks current numpy (2.0 ->
  2.5), polars (1.17 -> 1.44) and torch (2.7 -> 2.13).
- Modernised the code for 3.12: PEP 695 type parameters in place of explicit
  `TypeVar`s, and an explicit `strict=` on every `zip()`.

## [4.1.1] - 2026-08-24

### Fixed

- Export the recommender classes from the package root again. `from
  board_game_recommender import LightGamesRecommender` worked in v3 but raised
  `ImportError` in 4.0.0 and 4.1.0, since the package's `__init__.py` was empty.

### Changed

- API documentation now lists each class once, under the import path it is
  actually reachable from, rather than walking every submodule.

## [4.1.0] - 2026-08-24

### Added

- `evaluation` module for scoring recommenders: nDCG, exponential-gain nDCG, RMSE
  and effective catalog size, plus `ratings_train_test_split()` to hold out
  ratings from power users. Ported from v3 with no Turi Create or scikit-learn
  dependency; the effective catalog size formula is corrected (v3 computed
  `2 * sum(p * rank) + 1` where the definition is `- 1`, an offset of 2).
- `LightGamesRecommender.recommend_similar()` and `.similar_games()`, which were
  left unimplemented in 4.0.0. Both are back, now returning polars rather than
  pandas frames. Unlike in v3, an unknown game scores 0 against everything
  instead of returning `NaN` for every game.
- Document manual release process without GitHub Actions or `gh` CLI

## [4.0.0] - 2025-05-09

### Added

- Initial v4 implementation containing ABC, baseline and light recommenders

[Unreleased]: https://gitlab.com/recommend.games/board-game-recommender/compare/4.2.0...master
[4.2.0]: https://gitlab.com/recommend.games/board-game-recommender/compare/4.1.1...4.2.0
[4.1.1]: https://gitlab.com/recommend.games/board-game-recommender/compare/4.1.0...4.1.1
[4.1.0]: https://gitlab.com/recommend.games/board-game-recommender/compare/4.0.0...4.1.0
[4.0.0]: https://gitlab.com/recommend.games/board-game-recommender/tree/4.0.0
