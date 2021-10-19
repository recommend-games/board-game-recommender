"""Calculate the trust into users."""

import json

from itertools import groupby
from typing import Any, Dict, Generator, Iterable, Tuple, Union

import numpy as np
import pandas as pd

from scipy.stats import shapiro


def _user_trust(
    *,
    data: pd.DataFrame,
    min_ratings: int,
    ratings_col: str,
    date_col: str,
) -> float:
    if len(data) < min_ratings or ratings_col not in data or date_col not in data:
        return 0

    ratings = data[ratings_col].dropna()
    if len(ratings) < min_ratings:
        return 0

    if (ratings == ratings[0]).all():
        return 0

    months = (
        data[date_col].apply(lambda d: d[:7] if isinstance(d, str) else None).nunique()
    )
    if months < 2:
        return 0

    try:
        score, _ = shapiro(ratings)
    except Exception:
        return 0

    return score * np.log2(months)


def _users_trust(
    *,
    ratings: Union[str, Iterable[Dict[str, Any]]],
    min_ratings: int,
    key_col: str,
    ratings_col: str,
    date_col: str,
) -> Generator[Tuple[str, float], None, None]:
    if isinstance(ratings, str):
        with open(ratings, encoding="utf-8") as file:
            yield from _users_trust(
                ratings=map(json.loads, file),
                min_ratings=min_ratings,
                key_col=key_col,
                ratings_col=ratings_col,
                date_col=date_col,
            )
            return

    for key, group in groupby(ratings, key=lambda r: r[key_col]):
        data = pd.DataFrame.from_records(data=group)
        yield key, _user_trust(
            data=data,
            min_ratings=min_ratings,
            ratings_col=ratings_col,
            date_col=date_col,
        )


def user_trust(
    ratings: Union[str, Iterable[Dict[str, Any]]],
    *,
    min_ratings: int = 10,
    key_col: str = "bgg_user_name",
    ratings_col: str = "bgg_user_rating",
    date_col: str = "updated_at",
) -> pd.Series:
    """Calculate the trust in users."""

    trust = _users_trust(
        ratings=ratings,
        min_ratings=min_ratings,
        key_col=key_col,
        ratings_col=ratings_col,
        date_col=date_col,
    )
    data = pd.DataFrame.from_records(
        data=trust, columns=[key_col, "trust"], index=key_col
    )
    return data["trust"]
