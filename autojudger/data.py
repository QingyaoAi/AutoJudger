"""Task data loading.

Generalizes PRE's DataLoader: accepts a path (JSONL / JSON / CSV) or an inline
list of dicts, auto-detecting the format. Each task item is a dict whose keys
the judgment prompt template can reference via ``{{key}}``.
"""

from __future__ import annotations

import csv
import json
import os
from typing import List


class DataLoader:
    def __init__(self, source, fmt: str = "auto"):
        """source: a filesystem path or an already-loaded list of dicts.
        fmt: 'auto' | 'jsonl' | 'json' | 'csv'.
        """
        self.source = source
        self.fmt = fmt

    def get_task_items(self) -> List[dict]:
        # Inline list of dicts — used for programmatic / test calls.
        if isinstance(self.source, list):
            return [dict(item) for item in self.source]

        path = self.source
        if not os.path.exists(path):
            raise FileNotFoundError(f"Task data not found: {path}")

        fmt = self.fmt
        if fmt == "auto":
            ext = os.path.splitext(path)[1].lower()
            fmt = {".jsonl": "jsonl", ".json": "json", ".csv": "csv"}.get(ext, "jsonl")

        if fmt == "csv":
            with open(path, encoding="utf-8") as f:
                return [dict(row) for row in csv.DictReader(f, skipinitialspace=True)]

        if fmt == "json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("data", [data])
            return [dict(item) for item in data]

        # jsonl: one JSON object per line
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
