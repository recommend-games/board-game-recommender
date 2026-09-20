"""Tests for sorted-order export, memory-mapped loading and the eager set."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pytest

from board_game_recommender.light import (
    CollaborativeFilteringData,
    LabelSet,
    LightGamesRecommender,
    SortedLabelIndex,
    memmap_npz_members,
    savez_aligned,
)

if TYPE_CHECKING:
    from pathlib import Path

UNKNOWN_USER = "nobody"
UNKNOWN_GAME = 9999
NUM_USERS = 3


def _data(*, users_sorted: bool = False) -> CollaborativeFilteringData:
    """Three users deliberately *not* in label order, so sorting is observable."""
    return CollaborativeFilteringData(
        intercept=7.0,
        users_labels=np.array(["charlie", "alice", "bob"]),
        users_linear_terms=np.array([0.3, 0.1, 0.2]),
        users_factors=np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        items_labels=np.array([1, 2, 3]),
        items_linear_terms=np.array([0.1, 0.2, 0.3]),
        items_factors=np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        users_sorted=users_sorted,
    )


def test_to_npz_writes_users_in_sorted_label_order(tmp_path: Path) -> None:
    file_path = tmp_path / "model.npz"
    _data().to_npz(file_path)

    with file_path.open(mode="rb") as file:
        files = np.load(file=file)
        assert files["users_labels"].tolist() == ["alice", "bob", "charlie"]
        # the other user arrays move with the labels, not independently
        np.testing.assert_allclose(files["users_linear_terms"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(
            files["users_factors"],
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        )
        assert bool(files["users_sorted"]) is True


def test_to_npz_casts_factors_to_float32_on_disk(tmp_path: Path) -> None:
    # cast at export, not load: an `astype()` on load defeats mmap
    file_path = tmp_path / "model.npz"
    _data().to_npz(file_path)

    with file_path.open(mode="rb") as file:
        files = np.load(file=file)
        assert files["users_factors"].dtype == np.float32
        assert files["users_linear_terms"].dtype == np.float32
        assert files["items_factors"].dtype == np.float32
        assert files["items_linear_terms"].dtype == np.float32


def test_round_trip_preserves_scores_despite_reordering(tmp_path: Path) -> None:
    original = LightGamesRecommender(_data())
    file_path = tmp_path / "model.npz"
    original.to_npz(file_path)
    reloaded = LightGamesRecommender.from_npz(file_path)

    users = ["bob", "alice", "charlie"]
    np.testing.assert_allclose(
        reloaded.recommend_as_numpy(users, [1, 2, 3]),
        original.recommend_as_numpy(users, [1, 2, 3]),
        atol=1e-6,
    )


@pytest.mark.parametrize("mmap", [False, True])
def test_from_npz_scores_identically_with_and_without_mmap(
    tmp_path: Path,
    *,
    mmap: bool,
) -> None:
    file_path = tmp_path / "model.npz"
    _data().to_npz(file_path)

    recommender = LightGamesRecommender.from_npz(file_path, mmap=mmap)

    assert recommender.users_labels.tolist() == ["alice", "bob", "charlie"]
    np.testing.assert_allclose(
        recommender.recommend_as_numpy(["alice", "charlie"], [1, 2]),
        [
            [1.0 + 0.1 + 0.1 + 7.0, 0.0 + 0.1 + 0.2 + 7.0],
            [3.0 + 0.3 + 0.1 + 7.0, 0.0 + 0.3 + 0.2 + 7.0],
        ],
        atol=1e-6,
    )


def test_mmap_load_does_not_read_arrays_into_memory(tmp_path: Path) -> None:
    file_path = tmp_path / "model.npz"
    _data().to_npz(file_path)

    data = CollaborativeFilteringData.from_npz(file_path, mmap=True)

    assert isinstance(data.users_labels, np.memmap)
    assert isinstance(data.users_factors, np.memmap)
    assert isinstance(data.users_linear_terms, np.memmap)
    # item arrays stay in the heap: they are small and used in full every time
    assert not isinstance(data.items_factors, np.memmap)


def test_mmap_values_match_the_eager_load(tmp_path: Path) -> None:
    file_path = tmp_path / "model.npz"
    _data().to_npz(file_path)

    eager = CollaborativeFilteringData.from_npz(file_path)
    mapped = CollaborativeFilteringData.from_npz(file_path, mmap=True)

    np.testing.assert_array_equal(mapped.users_labels, eager.users_labels)
    np.testing.assert_array_equal(mapped.users_factors, eager.users_factors)
    np.testing.assert_array_equal(
        mapped.users_linear_terms,
        eager.users_linear_terms,
    )
    assert mapped.users_factors.dtype == np.float32


def test_mmap_load_skips_argsort_for_sorted_artefacts(tmp_path: Path) -> None:
    # `np.argsort()` would read every label, paging in the whole mapping
    file_path = tmp_path / "model.npz"
    _data().to_npz(file_path)

    recommender = LightGamesRecommender.from_npz(file_path, mmap=True)

    assert recommender.users_indexes.presorted


def test_unsorted_artefact_falls_back_to_eager_load(tmp_path: Path) -> None:
    # `searchsorted` over unsorted labels answers wrongly, and silently
    file_path = tmp_path / "legacy.npz"
    data = _data()
    with file_path.open(mode="wb") as file:
        np.savez(
            file=file,
            intercept=data.intercept,
            users_labels=data.users_labels,
            users_linear_terms=data.users_linear_terms,
            users_factors=data.users_factors,
            items_labels=data.items_labels,
            items_linear_terms=data.items_linear_terms,
            items_factors=data.items_factors,
        )

    loaded = CollaborativeFilteringData.from_npz(file_path, mmap=True)

    assert loaded.users_sorted is False
    assert not isinstance(loaded.users_factors, np.memmap)
    # and lookups still find every user, in the original order
    recommender = LightGamesRecommender(loaded)
    assert not recommender.users_indexes.presorted
    assert recommender.users_indexes[["charlie"]].tolist() == [0]


def test_memmap_npz_members_rejects_compressed_archives(tmp_path: Path) -> None:
    file_path = tmp_path / "compressed.npz"
    with file_path.open(mode="wb") as file:
        np.savez_compressed(file=file, users_labels=np.array(["alice", "bob"]))

    with pytest.raises(ValueError, match="compressed"):
        memmap_npz_members(file_path, ["users_labels"])


def test_savez_aligned_starts_every_member_on_an_aligned_offset(
    tmp_path: Path,
) -> None:
    # unaligned, numpy re-copies the whole array per operation: 18 ms and
    # all 55 MB paged in per lookup, against 38 us aligned
    file_path = tmp_path / "model.npz"
    # differing name lengths, so a fixed padding cannot accidentally work
    savez_aligned(
        file_path,
        a=np.array(["alice", "bob"]),
        longer_name=np.arange(9, dtype=np.float32),
        x=np.arange(5, dtype=np.int64),
    )

    mapped = memmap_npz_members(file_path, ["a", "longer_name", "x"])

    for name, array in mapped.items():
        assert array.flags.aligned, name
        assert array.__array_interface__["data"][0] % 64 == 0, name


def test_savez_aligned_is_readable_by_np_load(tmp_path: Path) -> None:
    file_path = tmp_path / "model.npz"
    labels = np.array(["alice", "bob"])
    savez_aligned(file_path, users_labels=labels, values=np.arange(3, dtype=np.float32))

    with file_path.open(mode="rb") as file:
        files = np.load(file=file)
        np.testing.assert_array_equal(files["users_labels"], labels)
        np.testing.assert_allclose(files["values"], [0.0, 1.0, 2.0])


def test_memmap_npz_members_rejects_unaligned_members(tmp_path: Path) -> None:
    file_path = tmp_path / "plain.npz"
    with file_path.open(mode="wb") as file:
        # `np.savez()` gives no control over member offsets
        np.savez(file=file, users_labels=np.array(["alice", "bob"]))

    mapped_or_error: object
    try:
        mapped_or_error = memmap_npz_members(file_path, ["users_labels"])
    except ValueError as error:
        assert "unaligned" in str(error)  # noqa: PT017
    else:
        # if np.savez happened to land it on an aligned offset, that is fine
        assert mapped_or_error["users_labels"].flags.aligned


def test_unknown_user_scores_as_the_average_baseline() -> None:
    # same result the appended zero row used to give
    recommender = LightGamesRecommender(_data())

    scores = recommender.recommend_as_numpy([UNKNOWN_USER], [1, 2, 3])

    np.testing.assert_allclose(scores, [[7.1, 7.2, 7.3]], atol=1e-12)


def test_unknown_user_does_not_borrow_the_last_users_factors() -> None:
    # regression: with the sentinel row gone, `factors[-1]` is a real user
    recommender = LightGamesRecommender(_data())

    known = recommender.recommend_as_numpy(["charlie"], [1])
    unknown = recommender.recommend_as_numpy([UNKNOWN_USER], [1])

    assert not np.allclose(known, unknown)


def test_unknown_game_scores_as_the_average_baseline() -> None:
    recommender = LightGamesRecommender(_data())

    scores = recommender.recommend_as_numpy(["alice"], [UNKNOWN_GAME])

    np.testing.assert_allclose(scores, [[7.1]], atol=1e-12)


def test_known_users_is_a_view_not_a_frozenset() -> None:
    recommender = LightGamesRecommender(_data())

    assert isinstance(recommender.known_users, LabelSet)
    assert "alice" in recommender.known_users
    assert UNKNOWN_USER not in recommender.known_users
    assert len(recommender.known_users) == NUM_USERS
    assert sorted(recommender.known_users) == ["alice", "bob", "charlie"]


def test_known_games_is_a_view_not_a_frozenset() -> None:
    recommender = LightGamesRecommender(_data())

    assert isinstance(recommender.known_games, LabelSet)
    assert 1 in recommender.known_games
    assert UNKNOWN_GAME not in recommender.known_games
    assert sorted(recommender.known_games) == [1, 2, 3]
    assert recommender.rated_games == recommender.known_games


def test_sorted_label_index_contains_scalar() -> None:
    index = SortedLabelIndex(np.array(["alice", "bob"]), presorted=True)

    assert "alice" in index
    assert UNKNOWN_USER not in index


def test_eager_users_are_copied_out_of_the_mapping(tmp_path: Path) -> None:
    file_path = tmp_path / "model.npz"
    _data().to_npz(file_path)

    recommender = LightGamesRecommender.from_npz(
        file_path,
        mmap=True,
        eager_users=["charlie", "alice"],
    )

    # sorted row indexes of alice (0) and charlie (2)
    assert recommender._eager_indexes.tolist() == [0, 2]  # noqa: SLF001
    assert not isinstance(recommender._eager_factors, np.memmap)  # noqa: SLF001


def test_eager_users_score_the_same_as_mapped_ones(tmp_path: Path) -> None:
    file_path = tmp_path / "model.npz"
    _data().to_npz(file_path)

    mapped = LightGamesRecommender.from_npz(file_path, mmap=True)
    eager = LightGamesRecommender.from_npz(
        file_path,
        mmap=True,
        eager_users=["alice", UNKNOWN_USER],
    )

    users = ["bob", "alice", UNKNOWN_USER, "charlie"]
    np.testing.assert_allclose(
        eager.recommend_as_numpy(users, [1, 2, 3]),
        mapped.recommend_as_numpy(users, [1, 2, 3]),
        atol=1e-6,
    )


def test_eager_users_ignores_names_the_artefact_does_not_know() -> None:
    # a premium list can move ahead of the artefact; that must not raise
    recommender = LightGamesRecommender(_data(), eager_users=[UNKNOWN_USER])

    assert recommender._eager_indexes.tolist() == []  # noqa: SLF001
    assert UNKNOWN_USER not in recommender.known_users


def test_sorted_export_keeps_every_user_paired_with_its_own_rows(
    tmp_path: Path,
) -> None:
    # a mis-paired permutation is undetectable corruption: everyone still
    # gets *a* recommendation, just somebody else's
    users = ["delta", "alpha", "echo", "bravo", "charlie"]
    # all distinct, so no axis confusion passes unnoticed
    factors = np.array(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [100.0, 200.0, 300.0],
            [1000.0, 2000.0, 3000.0],
            [10000.0, 20000.0, 30000.0],
        ],
    )
    data = CollaborativeFilteringData(
        intercept=0.0,
        users_labels=np.array(users),
        users_linear_terms=np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
        users_factors=factors,
        items_labels=np.array([1, 2]),
        items_linear_terms=np.array([0.0, 0.0]),
        items_factors=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
    )
    file_path = tmp_path / "model.npz"
    data.to_npz(file_path)

    reloaded = LightGamesRecommender.from_npz(file_path, mmap=True)

    for position, user in enumerate(users):
        np.testing.assert_allclose(
            reloaded.recommend_as_numpy([user], [1, 2]),
            [factors[position, :2] + data.users_linear_terms[position]],
            rtol=1e-6,
        )


def test_string_queries_against_int_labels_come_back_unknown() -> None:
    # game IDs arrive from URL params, so a string reaches this; narrowing
    # to the labels' dtype must miss, not raise
    recommender = LightGamesRecommender(_data())

    np.testing.assert_array_equal(
        recommender.items_indexes[["not-an-int", "2"]],
        [-1, -1],
    )
    # ...while actual ints are still found
    np.testing.assert_array_equal(recommender.items_indexes[[2]], [1])


def test_an_all_eager_batch_never_touches_the_mapping(tmp_path: Path) -> None:
    file_path = tmp_path / "model.npz"
    _data().to_npz(file_path)
    users = ["alice", "bob", "charlie"]

    recommender = LightGamesRecommender.from_npz(
        file_path,
        mmap=True,
        eager_users=users,
    )
    # a mapping that raises if read proves the eager copy is what serves
    recommender.users_factors = _ExplodingArray(recommender.users_factors)
    recommender.users_linear_terms = _ExplodingArray(recommender.users_linear_terms)

    scores = recommender.recommend_as_numpy(users, [1, 2, 3])

    assert scores.shape == (3, 3)


class _ExplodingArray(np.ndarray):
    """An array that fails loudly if anything reads an element of it."""

    def __new__(cls, source: np.ndarray) -> Self:
        return np.asarray(source).view(cls)

    def __getitem__(self, item: Any) -> Any:
        msg = f"the mapping was read: {item!r}"
        raise AssertionError(msg)
