"""
Data loader and feature engineering for the football analytics dashboard.

Default source:
    data/football.csv

This module intentionally contains no dashboard UI or charting code.
It exposes reusable functions for loading, validating, enriching, and
aggregating player statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


# Resolve the default CSV relative to the project root, not the shell's
# current working directory. This file is expected to live at the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "football.csv"

REQUIRED_COLUMNS = [
    "id",
    "player_name",
    "games",
    "time",
    "goals",
    "xG",
    "assists",
    "xA",
    "shots",
    "key_passes",
    "yellow_cards",
    "red_cards",
    "position",
    "team_title",
    "npg",
    "npxG",
    "xGChain",
    "xGBuildup",
]

NUMERIC_COLUMNS = [
    "id",
    "games",
    "time",
    "goals",
    "xG",
    "assists",
    "xA",
    "shots",
    "key_passes",
    "yellow_cards",
    "red_cards",
    "npg",
    "npxG",
    "xGChain",
    "xGBuildup",
]

COUNT_COLUMNS = [
    "games",
    "time",
    "goals",
    "assists",
    "shots",
    "key_passes",
    "yellow_cards",
    "red_cards",
    "npg",
]

# These fields can be summed across players to create team-level summaries.
# Rates and ratios should be calculated AFTER aggregation, never averaged
# directly from player-level rates.
TEAM_SUM_COLUMNS = [
    "games",
    "time",
    "goals",
    "xG",
    "assists",
    "xA",
    "shots",
    "key_passes",
    "yellow_cards",
    "red_cards",
    "npg",
    "npxG",
    "xGChain",
    "xGBuildup",
]


class FootballDataError(ValueError):
    """Raised when the football dataset cannot be used safely."""


    def _remove_export_index(df: pd.DataFrame) -> pd.DataFrame:
        """Remove a CSV-exported row index, such as the leading unnamed column."""
        unnamed = [
            col for col in df.columns
            if str(col).strip().lower().startswith("unnamed:")
        ]
        if unnamed:
            df = df.drop(columns=unnamed)
        return df


    def load_data(
        path: str | Path = DEFAULT_DATA_PATH,
        *,
        strict: bool = True,
        add_features: bool = True,
    ) -> pd.DataFrame:
        """
        Load and validate the player-level football CSV.

        Parameters
        ----------
        path:
            CSV path. Defaults to ``data/football.csv`` relative to this file.
        strict:
            If True, raise FootballDataError when required columns are missing.
            If False, return the loaded frame after basic column cleanup.
        add_features:
        """