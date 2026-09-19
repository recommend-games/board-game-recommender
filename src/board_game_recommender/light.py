"""Light recommender model, without the heavy Turi Create dependency."""

from __future__ import annotations

import logging
import struct
import zipfile
from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from board_game_recommender.abc import BaseGamesRecommender
from board_game_recommender.baseline import dataframe_from_scores

LOGGER = logging.getLogger(__name__)

# float32 precision exceeds the model's training noise, halving memory for free.
_FLOAT32_FIELDS = (
    "users_linear_terms",
    "users_factors",
    "items_linear_terms",
    "items_factors",
)

# The members served straight off the file rather than read into the heap.
_MMAP_FIELDS = (
    "users_labels",
    "users_linear_terms",
    "users_factors",
)

_ZIP_LOCAL_HEADER_SIZE = 30
_ZIP_LOCAL_HEADER_SIGNATURE = b"PK\x03\x04"

# Array data has to start at an aligned file offset, or numpy copies the whole
# array into an aligned buffer on every single operation -- see
# `_aligned_extra_field()`. 64 is what `.npy` itself pads its headers to.
_ARRAY_ALIGN = 64
# A .zip extra field is a sequence of (id, size, payload) records. 0xFFFF is
# not assigned to anything, so readers skip it; numpy's own loader included.
_PADDING_EXTRA_FIELD_ID = 0xFFFF
_EXTRA_FIELD_HEADER_SIZE = 4

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import IO, Self

    import polars as pl


class SortedLabelIndex:
    """Maps labels to their original position via a sorted permutation.

    An `np.searchsorted()` lookup over a permutation array avoids the
    per-item Python object overhead of a `dict` built from the same labels.

    `presorted=True` skips building the permutation at all: `np.argsort()`
    reads every element, which would page in a whole memory mapping.
    """

    def __init__(self, labels: np.ndarray, *, presorted: bool = False) -> None:
        # `sorter=` looks up positions without materializing a sorted copy of
        # `labels`, which -- unlike `self._order` -- is exactly as big as the
        # duplicate array this class exists to avoid.
        self._labels = labels
        self._order: np.ndarray | None = None if presorted else np.argsort(labels)

    @property
    def presorted(self) -> bool:
        """Whether the labels were already sorted, so no permutation exists."""
        return self._order is None

    def __getitem__(self, queries: Iterable[Any]) -> np.ndarray:
        # a bare str/int would otherwise iterate into garbage instead of erroring
        if isinstance(queries, (str, int, np.str_, np.integer)):
            msg = f"expected a batch of labels, got a single one: {queries!r}"
            raise TypeError(msg)
        queries_array = np.asarray(list(queries))
        if not queries_array.size:
            return np.empty(0, dtype=np.intp)
        positions = np.searchsorted(
            self._labels,
            self._as_labels_dtype(queries_array),
            sorter=self._order,
        ).clip(max=len(self._labels) - 1)
        order = self._order  # a local narrows where the `presorted` property can't
        candidate_indexes = positions if order is None else order[positions]
        found = self._labels[candidate_indexes] == queries_array
        return np.where(found, candidate_indexes, -1)

    def _as_labels_dtype(self, queries_array: np.ndarray) -> np.ndarray:
        """Narrow string queries to the labels' own width before searching.

        Two `<U` dtypes of different width make `searchsorted()` promote to a
        common one, copying the entire label array -- a full 55 MB scan where
        a ~20-page binary search was intended. Truncation is safe: `found`
        compares against the untruncated original. Only same-kind casts,
        since e.g. str-to-int64 raises where a miss is the right answer.
        """

        labels_dtype = self._labels.dtype
        if queries_array.dtype.kind != labels_dtype.kind or labels_dtype.kind not in (
            "U",
            "S",
        ):
            return queries_array
        return queries_array.astype(labels_dtype)

    def __contains__(self, label: Any) -> bool:
        """Membership for a single label, without materializing anything."""
        # `__getitem__` rejects scalars on purpose; this is the scalar door.
        return bool(self[[label]][0] >= 0)

    def __len__(self) -> int:
        return len(self._labels)

    def __iter__(self) -> Any:
        # Touches every label -- fine for a build-time `list()`, never on a
        # request path. Callers wanting membership should use `in` instead.
        return iter(self._labels.tolist())


class LabelSet(AbstractSet[Any]):
    """A set view over a `SortedLabelIndex`, answering `in` without a copy.

    The `frozenset(labels.tolist())` this replaces cost ~58 MB, and building
    it read every label -- which defeats a memory mapping outright.
    """

    def __init__(self, index: SortedLabelIndex) -> None:
        self._index = index

    def __contains__(self, label: object) -> bool:
        return label in self._index

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Any:
        return iter(self._index)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} labels)"


def _read_npy_member_layout(
    file: IO[bytes],
    header_offset: int,
) -> tuple[int, np.dtype, tuple[int, ...], bool]:
    """Locate one `.npy` member's array data inside an uncompressed `.npz`.

    Returns the byte offset of the raw array data, plus the dtype, shape and
    storage order parsed from the member's `.npy` header.
    """

    file.seek(header_offset)
    local_header = file.read(_ZIP_LOCAL_HEADER_SIZE)
    if local_header[:4] != _ZIP_LOCAL_HEADER_SIGNATURE:
        msg = f"no local file header at offset {header_offset}"
        raise ValueError(msg)
    # The local header's name/extra lengths are the authoritative ones: the
    # central directory's extra field legitimately differs in size from this.
    name_length, extra_length = struct.unpack("<HH", local_header[26:30])
    file.seek(header_offset + _ZIP_LOCAL_HEADER_SIZE + name_length + extra_length)

    version = np.lib.format.read_magic(file)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(file)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(file)
    else:
        msg = f"unsupported .npy format version {version}"
        raise ValueError(msg)

    return file.tell(), dtype, shape, fortran_order


def _aligned_extra_field(header_offset: int, filename: str) -> bytes:
    """Padding that pushes a zip member's payload to an aligned file offset.

    On an unaligned array numpy neither fails nor degrades: it copies the
    whole thing into an aligned buffer on *every* operation. That took one
    `searchsorted` over the mapped `users_labels` from 2.6 us to 18 ms and
    paged in all 55 MB, i.e. it silently undoes on-demand loading. `np.savez()`
    gives no control over member offsets, hence writing the archive by hand.

    numpy pads the member's own `.npy` header to 64, so aligning the payload
    aligns the array data with it.
    """

    payload_offset = (
        header_offset + _ZIP_LOCAL_HEADER_SIZE + len(filename.encode("utf-8"))
    )
    padding = -payload_offset % _ARRAY_ALIGN
    if padding == 0:
        return b""
    # A record cannot be shorter than its own 4-byte header.
    while padding < _EXTRA_FIELD_HEADER_SIZE:
        padding += _ARRAY_ALIGN
    return struct.pack(
        "<HH",
        _PADDING_EXTRA_FIELD_ID,
        padding - _EXTRA_FIELD_HEADER_SIZE,
    ) + bytes(padding - _EXTRA_FIELD_HEADER_SIZE)


def savez_aligned(file_path: Path | str, **arrays: Any) -> None:
    """Write an uncompressed `.npz` whose members are readable via `np.memmap`.

    Same output as `np.savez()`, except that every member's array data starts
    at a 64-byte-aligned offset in the file. `np.load()` reads it identically;
    `memmap_npz_members()` can additionally map it in place.
    """

    file_path = Path(file_path).resolve()
    with zipfile.ZipFile(
        file_path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for name, value in arrays.items():
            filename = f"{name}.npy"
            info = zipfile.ZipInfo(filename=filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.extra = _aligned_extra_field(archive.fp.tell(), filename)  # type: ignore[union-attr]
            # No `force_zip64`: its 20-byte extra field goes into the local
            # header too, which would undo the padding computed above.
            with archive.open(info, mode="w") as member:
                np.lib.format.write_array(
                    member,
                    np.asanyarray(value),
                    allow_pickle=False,
                )


def memmap_npz_members(
    file_path: Path | str,
    names: Iterable[str],
) -> dict[str, np.memmap]:
    """Memory-map the named members of an uncompressed `.npz` file in place.

    `np.load(..., mmap_mode="r")` **silently ignores** `mmap_mode` for `.npz`,
    so it cannot be used for this. Members are stored uncompressed, though, so
    each can be mapped out of the container at its own offset.
    """

    file_path = Path(file_path).resolve()
    mappings: dict[str, np.memmap] = {}

    with zipfile.ZipFile(file_path) as archive, file_path.open(mode="rb") as file:
        for name in names:
            info = archive.getinfo(f"{name}.npy")
            if info.compress_type != zipfile.ZIP_STORED:
                msg = (
                    f"member <{name}> is compressed and cannot be mapped; the "
                    "artefact must be written with np.savez(), not "
                    "np.savez_compressed()"
                )
                raise ValueError(msg)
            offset, dtype, shape, fortran_order = _read_npy_member_layout(
                file,
                info.header_offset,
            )
            mapping = np.memmap(
                file_path,
                dtype=dtype,
                shape=shape,
                order="F" if fortran_order else "C",
                mode="r",
                offset=offset,
            )
            if not mapping.flags.aligned:
                # Slower than not mapping at all -- see `_aligned_extra_field`.
                msg = (
                    f"member <{name}> starts at unaligned offset {offset}; "
                    "the artefact must be written by savez_aligned()"
                )
                raise ValueError(msg)
            mappings[name] = mapping

    return mappings


@dataclass(frozen=True)
class CollaborativeFilteringData:
    """Labels, vectors and matrices for linear collaborative filtering models."""

    intercept: float
    users_labels: np.ndarray  # (num_users,)
    users_linear_terms: np.ndarray  # (num_users,)
    users_factors: np.ndarray  # (num_users, num_factors)
    items_labels: np.ndarray  # (num_items,)
    items_linear_terms: np.ndarray  # (num_items,)
    items_factors: np.ndarray  # (num_factors, num_items)
    # Recorded rather than checked: verifying sortedness reads every label,
    # and assuming it wrongly answers every lookup wrongly, silently. Older
    # artefacts lack the member and load as unsorted.
    users_sorted: bool = False

    def _field_values(self) -> dict[str, Any]:
        # `dataclasses.asdict()` would deep-copy all 226 MB just to save it.
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def sorted_by_user_label(self) -> Self:
        """The same data with the user arrays in ascending label order.

        This is what lets `np.searchsorted()` run against a mapped label array
        with no permutation to build at load time.
        """

        if self.users_sorted:
            return self

        order = np.argsort(self.users_labels)
        return replace(
            self,
            users_labels=self.users_labels[order],
            users_linear_terms=self.users_linear_terms[order],
            users_factors=self.users_factors[order],
            users_sorted=True,
        )

    def to_npz(self, file_path: Path | str) -> None:
        """Save data into an .npz file."""

        file_path = Path(file_path).resolve()
        LOGGER.info("Saving data as .npz to <%s>", file_path)
        values = self.sorted_by_user_label()._field_values()  # noqa: SLF001
        # Not on load: an `astype()` there materializes the whole array.
        for key in _FLOAT32_FIELDS:
            values[key] = values[key].astype(np.float32, copy=False)
        savez_aligned(file_path, **values)
        LOGGER.info("Done saving <%s>", file_path)

    @classmethod
    def from_npz(cls, file_path: Path | str, *, mmap: bool = False) -> Self:
        """Load data from an .npz file.

        `mmap=True` serves the user arrays off the file, so only the pages a
        lookup touches become resident. An artefact not in sorted user order
        is loaded eagerly instead, since `searchsorted` would otherwise lie.
        """

        file_path = Path(file_path).resolve()
        LOGGER.info("Loading data as .npz from <%s>", file_path)
        with file_path.open(mode="rb") as file:
            files = np.load(file=file)
            keys = list(files.files)
            users_sorted = "users_sorted" in keys and bool(files["users_sorted"])

            if mmap and not users_sorted:
                LOGGER.warning(
                    "Artefact <%s> is not in sorted user order, so it can't be "
                    "memory-mapped; loading it into memory instead. Re-export "
                    "it to enable on-demand loading.",
                    file_path,
                )

            mapped = (
                memmap_npz_members(file_path, _MMAP_FIELDS)
                if mmap and users_sorted
                else {}
            )

            values: dict[str, Any] = {}
            for key in keys:
                if key == "users_sorted":
                    values[key] = users_sorted
                elif key == "intercept":
                    values[key] = float(files[key])
                elif key in mapped:
                    values[key] = mapped[key]
                elif key in _FLOAT32_FIELDS:
                    # Pre-4.7 artefacts still arrive as float64.
                    values[key] = files[key].astype(np.float32, copy=False)
                else:
                    values[key] = files[key]

            assert all(
                isinstance(key, str) and isinstance(value, (np.ndarray, float, bool))
                for key, value in values.items()
            ), "All keys must be strings and all values arrays, floats or bools"
            return cls(**values)


def _gather(vector: np.ndarray, indexes: np.ndarray) -> np.ndarray:
    """`vector[indexes]`, with a zero in place of every `-1` (unknown label)."""

    known = indexes >= 0
    result = np.zeros(len(indexes), dtype=vector.dtype)
    if known.any():
        result[known] = vector[indexes[known]]
    return result


def _gather_rows(matrix: np.ndarray, indexes: np.ndarray) -> np.ndarray:
    """`matrix[indexes]`, with a zero row in place of every `-1`."""

    known = indexes >= 0
    result = np.zeros((len(indexes), matrix.shape[1]), dtype=matrix.dtype)
    if known.any():
        result[known] = matrix[indexes[known]]
    return result


def _gather_columns(matrix: np.ndarray, indexes: np.ndarray) -> np.ndarray:
    """`matrix[:, indexes]`, with a zero column in place of every `-1`."""

    known = indexes >= 0
    result = np.zeros((matrix.shape[0], len(indexes)), dtype=matrix.dtype)
    if known.any():
        result[:, known] = matrix[:, indexes[known]]
    return result


class LightGamesRecommender(BaseGamesRecommender[int, str]):
    """Light recommender without Turi Create dependency."""

    def __init__(
        self,
        data: CollaborativeFilteringData,
        *,
        eager_users: Iterable[str] | None = None,
    ) -> None:
        """
        `eager_users` are read into the heap up front so they never pay for a
        page fault. Unknown names are ignored, so a premium-user list is safe
        to pass even when it has moved ahead of the artefact.
        """

        assert data.users_factors.shape[-1] == data.items_factors.shape[0]
        # TODO check other dimensions as well (num_users and num_items)

        self.intercept: float = data.intercept

        self.users_labels: np.ndarray = data.users_labels
        self.users_indexes = SortedLabelIndex(
            data.users_labels,
            presorted=data.users_sorted,
        )
        # Stored unpadded: the zero sentinel row this used to append had to
        # materialize the whole array. `_gather*()` resolves misses instead.
        self.users_linear_terms = data.users_linear_terms
        self.users_factors = data.users_factors

        self.items_labels: np.ndarray = data.items_labels
        self.items_indexes = SortedLabelIndex(data.items_labels)
        self.items_linear_terms = data.items_linear_terms
        self.items_factors = data.items_factors

        self._known_games: AbstractSet[int] = LabelSet(self.items_indexes)
        self._known_users: AbstractSet[str] = LabelSet(self.users_indexes)

        self._eager_indexes = np.empty(0, dtype=np.intp)
        self._eager_factors = np.empty(
            (0, self.users_factors.shape[1]),
            dtype=self.users_factors.dtype,
        )
        self._eager_linear_terms = np.empty(
            0,
            dtype=self.users_linear_terms.dtype,
        )
        if eager_users is not None:
            self._load_eager_users(eager_users)

        LOGGER.info(
            "Loaded light recommender with %d users (%d of them eagerly) and %d items",
            len(self.users_labels),
            len(self._eager_indexes),
            len(self.items_labels),
        )

    def _load_eager_users(self, eager_users: Iterable[str]) -> None:
        """Copy the given users' rows into the heap, so lookups never fault."""

        indexes = self.users_indexes[list(eager_users)]
        # `unique()` also sorts, which is what `_user_rows()` searches against.
        self._eager_indexes = np.unique(indexes[indexes >= 0])
        # `asarray()` copies out of the mapping; that is the point.
        self._eager_factors = np.asarray(self.users_factors[self._eager_indexes])
        self._eager_linear_terms = np.asarray(
            self.users_linear_terms[self._eager_indexes],
        )

    def _user_rows(self, user_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Factors and linear terms for user rows, preferring the eager copy."""

        if not len(self._eager_indexes):
            return (
                _gather_rows(self.users_factors, user_ids),
                _gather(self.users_linear_terms, user_ids),
            )

        positions = np.searchsorted(self._eager_indexes, user_ids).clip(
            max=len(self._eager_indexes) - 1,
        )
        eager = self._eager_indexes[positions] == user_ids

        # Masked out of the mapped read, then filled from the heap copy, so a
        # resident user never touches the mapping at all.
        mapped_ids = np.where(eager, -1, user_ids)
        factors = _gather_rows(self.users_factors, mapped_ids)
        linear_terms = _gather(self.users_linear_terms, mapped_ids)
        if eager.any():
            factors[eager] = self._eager_factors[positions[eager]]
            linear_terms[eager] = self._eager_linear_terms[positions[eager]]
        return factors, linear_terms

    def to_npz(self, file_path: Path | str) -> None:
        """Save data into an .npz file."""
        CollaborativeFilteringData(
            intercept=self.intercept,
            users_labels=self.users_labels,
            users_linear_terms=self.users_linear_terms,
            users_factors=self.users_factors,
            items_labels=self.items_labels,
            items_linear_terms=self.items_linear_terms,
            items_factors=self.items_factors,
            users_sorted=self.users_indexes.presorted,
        ).to_npz(file_path)

    @classmethod
    def from_npz(
        cls,
        file_path: Path | str,
        *,
        mmap: bool = False,
        eager_users: Iterable[str] | None = None,
    ) -> Self:
        """Load data from an .npz file."""
        data = CollaborativeFilteringData.from_npz(file_path, mmap=mmap)
        return cls(data, eager_users=eager_users)

    @property
    def known_games(self) -> AbstractSet[int]:
        return self._known_games

    @property
    def rated_games(self) -> AbstractSet[int]:
        return self.known_games

    @property
    def num_games(self) -> int:
        return len(self.items_labels)

    @property
    def known_users(self) -> AbstractSet[str]:
        return self._known_users

    @property
    def num_users(self) -> int:
        return len(self.users_labels)

    def _recommendation_scores(
        self,
        *,
        users: list[str] | None = None,
        games: list[int] | None = None,
        avg_users: bool = False,
    ) -> np.ndarray:
        """Calculate recommendations scores for certain users and games."""

        if users:
            user_ids = self.users_indexes[users]
            users_factors, users_linear_terms_1d = self._user_rows(user_ids)
            users_linear_terms = users_linear_terms_1d.reshape(-1, 1)
        else:
            # Reads every factor off disk when mapped; no request path does
            # this, they all pass explicit users.
            LOGGER.warning(
                "Scoring all %d users at once; this reads every user factor",
                self.num_users,
            )
            users_factors = self.users_factors
            users_linear_terms = self.users_linear_terms.reshape(-1, 1)

        if avg_users:
            users_factors = users_factors.mean(axis=0).reshape(1, -1)
            users_linear_terms = users_linear_terms.mean(axis=0).reshape(1, 1)

        if games:
            game_ids = self.items_indexes[games]
            items_factors = _gather_columns(self.items_factors, game_ids)
            items_linear_terms = _gather(self.items_linear_terms, game_ids).reshape(
                1,
                -1,
            )
        else:
            items_factors = self.items_factors
            items_linear_terms = self.items_linear_terms.reshape(1, -1)

        return cast(
            "np.ndarray",
            users_factors @ items_factors  # (num_users, num_items)
            + users_linear_terms  # (num_users, 1)
            + items_linear_terms  # (1, num_items)
            + self.intercept,  # (1,)
        )

    def _game_scores(
        self,
        games: list[int] | None = None,
    ) -> np.ndarray:
        """Calculate average game scores from bias terms."""

        if games:
            game_ids = self.items_indexes[games]
            items_linear_terms = _gather(self.items_linear_terms, game_ids)
        else:
            items_linear_terms = self.items_linear_terms

        return cast("np.ndarray", items_linear_terms + self.intercept)

    def recommend(
        self,
        users: Iterable[str],
        **kwargs: Any,  # noqa: ARG002
    ) -> pl.DataFrame:
        """Calculate recommendations for certain users."""

        users = list(users)
        scores = self._recommendation_scores(users=users)
        return dataframe_from_scores(
            users=users,
            games=self.items_labels,
            scores=scores,
        )

    def recommend_as_numpy(
        self,
        users: Iterable[str],
        games: Iterable[int],
    ) -> np.ndarray:
        """Calculate recommendations for certain users and games as a numpy array."""

        users = list(users)
        games = list(games)

        return self._recommendation_scores(users=users, games=games)

    def recommend_group(
        self,
        users: Iterable[str],
        **kwargs: Any,  # noqa: ARG002
    ) -> pl.DataFrame:
        """Calculate recommendations for a group of users."""

        users = list(users)
        scores = (
            self._recommendation_scores(users=users, avg_users=True)
            if users
            else self._game_scores()
        )
        return dataframe_from_scores(
            users=["_all"],
            games=self.items_labels,
            scores=scores,
        )

    def recommend_group_as_numpy(
        self,
        users: Iterable[str],
        games: Iterable[int],
    ) -> np.ndarray:
        """
        Calculate recommendations for a group of users and games as a numpy array.
        """

        users = list(users)
        games = list(games)
        return (
            self._recommendation_scores(users=users, games=games, avg_users=True)
            if users
            else self._game_scores(games).reshape(1, -1)
        )

    def _similarity_scores(self, games: list[int]) -> np.ndarray:
        """
        Cosine similarity between the given games and every known game.

        The result has shape (len(games), num_games). Games unknown to the model
        have no latent factors, so they score 0 against everything.
        """

        game_ids = self.items_indexes[games]
        game_factors = _gather_columns(self.items_factors, game_ids)
        return cosine_similarity(game_factors, self.items_factors)

    def recommend_similar(
        self,
        games: Iterable[int],
        **kwargs: Any,  # noqa: ARG002
    ) -> pl.DataFrame:
        """
        Recommend games similar to the given games based on cosine similarity
        of latent factors, averaged over all the given games.
        """

        games = list(games)
        scores = (
            self._similarity_scores(games).mean(axis=0).reshape(1, -1)
            if games
            else np.zeros((1, self.num_games))
        )
        return dataframe_from_scores(
            users=["_all"],
            games=self.items_labels,
            scores=scores,
        )

    def similar_games(
        self,
        games: Iterable[int],
        **kwargs: Any,  # noqa: ARG002
    ) -> pl.DataFrame:
        """
        Find games similar to the given games based on
        cosine similarity of latent factors.
        """

        games = list(games)
        return dataframe_from_scores(
            users=games,  # type: ignore[arg-type]
            games=self.items_labels,
            scores=self._similarity_scores(games),
        )


def cosine_similarity(matrix_1: np.ndarray, matrix_2: np.ndarray) -> np.ndarray:
    """
    Calculates the cosine similarity between two matrices.

    The input matrices need to be of shape (m,n) and (m,l);
    the result shape will be (n,l). Columns with zero norm score 0 throughout.
    """

    dot_product = matrix_1.T @ matrix_2  # (n,l)
    matrix_1_norm = np.linalg.norm(matrix_1, axis=0)  # (n,)
    matrix_2_norm = np.linalg.norm(matrix_2, axis=0)  # (l,)
    outer_prod_norm = np.outer(matrix_1_norm, matrix_2_norm)  # (n,l)

    # Zero vectors are orthogonal to everything by convention, rather than NaN:
    # a single unknown game would otherwise poison an entire recommendation.
    return cast(
        "np.ndarray",
        np.divide(
            dot_product,
            outer_prod_norm,
            out=np.zeros_like(dot_product),
            where=outer_prod_norm != 0,
        ),
    )  # (n,l)
