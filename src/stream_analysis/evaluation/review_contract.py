"""Shared identity contract for annotation and evaluation reviewers."""

from __future__ import annotations

import re


CANONICAL_REVIEWER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def canonical_reviewer_id(value: object, context: str = "reviewer_id") -> str:
    """Validate and return one lowercase reviewer identifier.

    Coordination and final benchmark publication share this function so an
    identity accepted at one stage cannot fail only at a later stage.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    if value != value.casefold() or not CANONICAL_REVIEWER_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{context} must be a canonical lowercase reviewer ID containing only "
            "letters, digits, underscore, or hyphen."
        )
    return value
