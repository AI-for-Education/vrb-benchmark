"""Shared utility functions for benchmark analysis scripts."""

import re

import pandas as pd


def tokenize_skills(skills_str: str) -> list[str]:
    """Tokenize skills string into a list of individual skills."""
    if pd.isna(skills_str):
        return []
    return [s.strip() for s in re.split(r"[;,]", skills_str) if s.strip()]
