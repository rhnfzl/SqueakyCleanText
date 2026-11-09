"""Placeholder syntax shared by cleaning, entity detection and scoring."""

import re

LABEL_PATTERN = r"[A-Z][A-Z0-9_]*"
PLACEHOLDER_PATTERN = rf"<{LABEL_PATTERN}>"
PLACEHOLDER_TOKEN = re.compile(PLACEHOLDER_PATTERN)
