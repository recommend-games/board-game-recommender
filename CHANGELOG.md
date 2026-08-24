# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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

[Unreleased]: https://gitlab.com/recommend.games/board-game-recommender/compare/4.1.0...master
[4.1.0]: https://gitlab.com/recommend.games/board-game-recommender/compare/4.0.0...4.1.0
[4.0.0]: https://gitlab.com/recommend.games/board-game-recommender/tree/4.0.0
