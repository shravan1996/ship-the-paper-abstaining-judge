"""Write data/items.jsonl and data/corpus.jsonl from the reference tables.

The output depends on the machine's /etc/services, tzdata and Python's Unicode version, so the
files used for the article are committed. Rebuilding on another system can change a few items.
"""

import json
import platform
import unicodedata
from pathlib import Path

from judge.facts import build_all

ROOT = Path(__file__).parent


def main() -> None:
    items, docs = build_all()
    (ROOT / "data").mkdir(exist_ok=True)
    with open(ROOT / "data" / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    with open(ROOT / "data" / "corpus.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    tz = Path("/usr/share/zoneinfo/+VERSION")
    meta = {
        "items": len(items),
        "corpus_documents": len(docs),
        "by_family": {f: sum(1 for it in items if it["family"] == f) for f in sorted({it["family"] for it in items})},
        "python": platform.python_version(),
        "unicode": unicodedata.unidata_version,
        "tzdata": tz.read_text().strip() if tz.exists() else "unknown",
        "platform": platform.platform(),
    }
    (ROOT / "data" / "build_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
