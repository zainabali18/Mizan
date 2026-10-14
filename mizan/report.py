"""Turn results/*.json into the Markdown results table for the README.

Run:  python -m mizan.report          (prints the table)
      python -m mizan.report --write  (also replaces the table in README.md)
"""
from __future__ import annotations

import json
import sys

from .evaluate import RESULTS, ROOT

START, END = "<!-- results:start -->", "<!-- results:end -->"
ORDER = ["TF-IDF + logistic regression", "Zero-shot LLM (Qwen2.5-1.5B-Instruct)", "Fine-tuned AraBERT"]


def pct(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:.1f}%"


def table() -> str:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("*.json"))]
    rows.sort(key=lambda r: ORDER.index(r["model"]) if r["model"] in ORDER else len(ORDER))
    lines = [
        "| Model | Accuracy | Macro-F1 | MSA F1 | Dialect (PAL) F1 | Latency / query | Size |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        v = r.get("by_variety", {})
        lines.append(
            f"| {r['model']} | {pct(r['accuracy'])} | **{pct(r['macro_f1'])}** | "
            f"{pct(v.get('MSA', {}).get('macro_f1'))} | {pct(v.get('PAL', {}).get('macro_f1'))} | "
            f"{r['latency_ms']:.1f} ms | {r['size_mb']:.1f} MB |"
        )
    return "\n".join(lines)


def main() -> None:
    md = table()
    print(md)
    if "--write" in sys.argv:
        readme = ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")
        if START not in text or END not in text:
            raise SystemExit("README is missing the results markers")
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        readme.write_text(f"{before}{START}\n{md}\n{END}{after}", encoding="utf-8")
        print("README.md updated")


if __name__ == "__main__":
    main()
