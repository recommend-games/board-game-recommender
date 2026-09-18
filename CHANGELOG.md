# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `LightGamesRecommender.from_npz()` now loads `users_factors`,
  `items_factors`, `users_linear_terms`, and `items_linear_terms` as
  `float32` instead of `float64`, roughly halving the in-memory footprint of
  a loaded model. Files on disk are unaffected — `to_npz()` still writes
  whatever dtype the data was constructed with.

## [4.5.0] - 2026-09-15

### Added

- `early_stopping_callback()` in the `dnn` module: previously private, now
  public and importable directly from `board_game_recommender.dnn`, for a
  caller orchestrating training in-process (e.g. a build task that already
  has ratings loaded) rather than through the CLI. It now returns the
  callback together with an `EarlyStoppingState` tracking best value, best
  epoch, and whether it fired, so the outcome can be inspected after
  training instead of only appearing in logs.
- `split_train_test()` and `recommender_test_data_from_frame()` in the
  `evaluation` module: the DataFrame-facing halves of
  `ratings_train_test_split()` and `load_test_data()`, for splitting and
  evaluating ratings already held in memory without a file round trip.
  `ratings_train_test_split()` and `load_test_data()` are now thin
  file-reading/writing wrappers around them.
- `train()`'s `on_epoch_end` now also accepts an iterable of callbacks, run
  together every epoch (stopping if any of them do), so combining early
  stopping with checkpointing or metadata capture no longer needs a private
  helper.
- `train()` accepts an optional `lr_scheduler_factory`, called once with the
  `Adam` optimizer to build a `torch.optim.lr_scheduler`, `.step()`'d at the
  end of every epoch. Left unset, the learning rate stays flat as before.
  The CLI exposes step decay directly via `--lr-step-size` and `--lr-gamma`.
- `training_metadata()` and `write_training_metadata()` in the `dnn` module.
  `training_metadata()` builds a provenance dict recording every
  hyperparameter actually used, the early-stopping outcome if early stopping
  was used, and the installed library version; `write_training_metadata()`
  persists a dict as a JSON sidecar next to a trained model's `.npz` (e.g.
  `model.npz` -> `model.json`). Kept as two functions so a calling
  application (e.g. a build task) can enrich the dict with
  deployment-specific facts it holds and the library can't know -- a git
  SHA, the identity of the data snapshot trained on -- before persisting it.
  The CLI calls both, unenriched, automatically after training.

### Changed

- `TrainingResult` gained a required `unobserved_rating_value` field, so the
  value `train()` actually fit against (resolved from the data when left
  unset) is recoverable afterwards instead of only appearing in a log line.
  **Breaking** for any code constructing `TrainingResult` directly.

## [4.4.0] - 2026-08-30

### Added

- `train()`'s `on_epoch_end` callback can now request an early stop by
  returning a truthy value. The CLI exposes this as
  `--early-stopping-metric` (any `RecommenderMetrics` field, e.g. `rmse`,
  `ndcg`, `catalog_coverage`), `--early-stopping-patience`, and
  `--early-stopping-eval-every`. Restores the best weights seen, not just
  whatever the last non-improving epoch produced. RMSE is a valid choice
  but not the default one to reach for: it can get slightly worse while
  ranking/diversity metrics keep improving, so picking the metric that
  matches what more training is meant to buy is on the caller.
- `train()` takes an `on_epoch_end` callback, called after every epoch with
  the model trained so far. `python -m board_game_recommender.dnn` exposes
  this as `--checkpoint-every N`, saving the model alongside the final
  output every N epochs, for inspecting a long run's trajectory instead of
  only its final result.
- `catalog_coverage()`: the fraction of games seen in the test data that
  appear in anyone's top-k, for every top-k cutoff. Complements
  `effective_catalog_size()`, which can look fine even when most of the
  candidate pool is never recommended to anyone, as long as whatever is
  recommended is spread evenly.
- `novelty()`: mean self-information of the top-k recommendations, using
  each game's frequency across all users' test rows as a popularity proxy.
  Both are now part of `calculate_metrics()` and the CLI's logged output.

## [4.3.0] - 2026-08-28

### Added

- `train()` in the `dnn` module: a plain training loop for
  `CollaborativeFilteringModel`, minimising mean squared error with `Adam`
  over shuffled minibatches. Intercept and biases start from the data's
  global and per-user/item mean ratings rather than 0, which matters a lot
  in practice: on a 1.1M-row sample of real BGG ratings, RMSE after 10
  untuned epochs went from 6.4 to 1.3.
- `train()` now also matches Turi's `RankingFactorizationRecommender`
  objective, not just plain MSE: `regularization`/`linear_regularization`
  (L2 on factors/biases) and `ranking_regularization` (for each row, sample
  `num_sampled_negative_examples` items for that user, push the
  highest-scoring one towards `unobserved_rating_value`), all defaulting to
  Turi's own values. `ranking_regularization=0` disables the ranking term.
- `TrainingResult.to_collaborative_filtering_data()`, converting a trained
  model into what `LightGamesRecommender` serves. Chains with the existing
  `.to_npz()` to save a trained model.
- `python -m board_game_recommender.dnn ratings.jl model.npz`: splits off a
  held-out sample of power users, trains, logs RMSE/nDCG/ECS against it, and
  saves the result. Exposes every `train()` hyperparameter as a flag.

### Fixed

- A rating with a missing game or user id crashed `train()`. Rows with a
  missing game or user id are now dropped, same as rows with a missing
  rating already were.
- `calculate_metrics()` no longer reports effective catalog size (ECS) at the
  test set's full width. At that cutoff every candidate is "recommended" to
  every user, so ECS collapses to a property of the test split rather than
  the model's ranking, silently producing a meaningless number whenever a
  caller's requested `k` happened to equal that width.
- `mypy --strict` was silently disabled project-wide: a leftover template
  override in `pyproject.toml` applied `ignore_missing_imports`/
  `implicit_reexport` with no `module` filter, matching everything. Removed
  it, added a properly scoped override for the optional `torch` import, and
  fixed the 24 type errors it had been hiding.

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

[Unreleased]: https://gitlab.com/recommend.games/board-game-recommender/compare/4.5.0...master
[4.5.0]: https://gitlab.com/recommend.games/board-game-recommender/compare/4.4.0...4.5.0
[4.4.0]: https://gitlab.com/recommend.games/board-game-recommender/compare/4.3.0...4.4.0
[4.3.0]: https://gitlab.com/recommend.games/board-game-recommender/compare/4.2.0...4.3.0
[4.2.0]: https://gitlab.com/recommend.games/board-game-recommender/compare/4.1.1...4.2.0
[4.1.1]: https://gitlab.com/recommend.games/board-game-recommender/compare/4.1.0...4.1.1
[4.1.0]: https://gitlab.com/recommend.games/board-game-recommender/compare/4.0.0...4.1.0
[4.0.0]: https://gitlab.com/recommend.games/board-game-recommender/tree/4.0.0
