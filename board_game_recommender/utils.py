# -*- coding: utf-8 -*-

"""Utility functions."""

import csv
import logging
import os
import sys

import turicreate as tc

csv.field_size_limit(sys.maxsize)

LOGGER = logging.getLogger(__name__)


def condense_csv(in_file, out_file, columns, header=True):
    """copying only columns from in_file to out_file"""

    if isinstance(in_file, str):
        with open(in_file, encoding="utf-8") as in_file_obj:
            return condense_csv(in_file_obj, out_file, columns)

    if isinstance(out_file, str):
        with open(out_file, "w", encoding="utf-8") as out_file_obj:
            return condense_csv(in_file, out_file_obj, columns)

    columns = tuple(columns)

    reader = csv.DictReader(in_file)
    writer = csv.DictWriter(out_file, columns)

    if header:
        writer.writeheader()

    count = -1

    for count, item in enumerate(reader):
        writer.writerow({k: item.get(k) for k in columns})

    return count + 1


def filter_sframe(sframe, **params):
    """query an SFrame with given parameters"""

    if not params:
        return sframe

    ind = tc.SArray.from_const(True, len(sframe))

    for key, value in params.items():
        split = key.split("__")
        if len(split) == 1:
            split.append("exact")
        field, operation = split

        sarray = sframe[field]

        if operation == "exact":
            ind &= sarray == value
        elif operation == "iexact":
            value = value.lower()
            ind &= sarray.apply(str.lower) == value
        elif operation == "ne":
            ind &= sarray != value
        elif operation == "contains":
            ind &= sarray.apply(lambda string, v=value: v in string)
        elif operation == "icontains":
            value = value.lower()
            ind &= sarray.apply(lambda string, v=value: v in string.lower())
        elif operation == "in":
            value = frozenset(value)
            ind &= sarray.apply(lambda item, v=value: item in v)
        elif operation == "gt":
            ind &= sarray > value
        elif operation == "gte":
            ind &= sarray >= value
        elif operation == "lt":
            ind &= sarray < value
        elif operation == "lte":
            ind &= sarray <= value
        elif operation == "range":
            lower, upper = value
            ind &= (sarray >= lower) & (sarray <= upper)
        elif operation == "apply":
            ind &= sarray.apply(value)
        else:
            raise ValueError(f"unknown operation <{operation}>")

    return sframe[ind]


def percentile_buckets(sarray, percentiles):
    """make percentiles"""

    sarray = sarray.sort(True)
    total = len(sarray)

    if not total:
        return

    percentiles = list(percentiles)
    assert percentiles == sorted(percentiles)
    percentiles = (
        [p / 100 for p in percentiles] if max(percentiles) > 1 else percentiles
    )
    assert 0 < max(percentiles) < 1
    percentiles.append(1)

    lower = sarray[0]

    for percentile in percentiles:
        index = int(percentile * total) if percentile < 1 else -1
        upper = sarray[index]
        LOGGER.debug(
            "%5.1f%%-tile: between %.3f and %.3f", percentile * 100, lower, upper
        )
        yield percentile, lower, upper
        lower = upper


def star_rating(score, buckets, low=1, high=5):
    """star rating"""

    if not buckets or len(buckets) < 2:
        return None

    step = (high - low) / (len(buckets) - 1)
    for i, (_, _, upper) in enumerate(buckets):
        if score <= upper:
            return low + i * step
    return high


def format_from_path(path):
    """get file extension"""
    try:
        _, ext = os.path.splitext(path)
        return ext.lower()[1:] if ext else None
    except Exception:
        pass
    return None
