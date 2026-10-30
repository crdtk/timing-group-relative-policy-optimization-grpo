#!/usr/bin/env python3
"""Draw docs/stage-breakdown.svg from the CSV files in results/.

One bar per run. Each bar splits the time of one training step into its stages, as a share of
the whole. Step 1 of every run is left out, because it pays for warm-up. Standard library only.

Usage: python3 make_chart.py
"""

import csv
import glob
import statistics
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = [("generation", "#2a78d6", "the model writes its answers"), ("backward", "#eb6834", "gradients are computed"),
          ("logprob", "#1baf7a", "the model scores its own answers"), ("optimizer", "#7a5cc7", "the weights change"),
          ("other", "#a8a7a1", "everything else"), ("scoring", "#d4d3cd", "the checker marks each answer")]
W, LEFT, BAR, ROW = 900, 250, 610, 46


def shares(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    use = rows[1:] if len(rows) > 2 else rows
    wall = statistics.mean(float(r["wall_s"]) for r in use)
    out = {s: 100 * statistics.mean(float(r[f"{s}_s"]) for r in use) / wall for s, _, _ in STAGES}
    meta = rows[0]
    label = f"{meta['gpu'].replace('Tesla-', '')}, {meta['script']}, batch {meta['batch']}"
    return label, meta["run"], out, wall


def main():
    runs = [shares(p) for p in sorted(glob.glob(str(HERE / "results" / "*.csv")))]
    if not runs:
        raise SystemExit("no CSV in results/")
    height = 120 + ROW * len(runs) + 40 + 22 * len(STAGES)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
           f'<rect width="{W}" height="{height}" fill="#fcfcfb"/>',
           '<text x="30" y="44" font-size="22" font-weight="700" fill="#0b0b0b">Where one training step spends its time</text>',
           '<text x="30" y="70" font-size="14" fill="#52514e">Qwen3 0.6B learning with Group Relative Policy Optimization on a Google Colab T4. Each bar is 100% of one step, averaged over the steps of one run.</text>']
    y = 110
    for label, run, out, wall in runs:
        svg.append(f'<text x="30" y="{y + 16}" font-size="14" font-weight="600" fill="#0b0b0b">{escape(label)}</text>')
        svg.append(f'<text x="30" y="{y + 34}" font-size="12" fill="#6b6a66">run {escape(run)}, {wall:.1f} s a step</text>')
        x = LEFT
        for s, colour, _ in STAGES:
            w = BAR * out[s] / 100
            if w < 0.5:
                continue
            svg.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="30" fill="{colour}"/>')
            if out[s] >= 6:
                svg.append(f'<text x="{x + w / 2:.1f}" y="{y + 20}" font-size="14" font-weight="700" text-anchor="middle" fill="#ffffff">{out[s]:.0f}%</text>')
            x += w
        y += ROW
    y += 18
    mean = {s: statistics.mean(r[2][s] for r in runs) for s, _, _ in STAGES}
    for s, colour, what in STAGES:
        svg.append(f'<rect x="30" y="{y - 12}" width="14" height="14" fill="{colour}"/>')
        svg.append(f'<text x="52" y="{y}" font-size="13" fill="#0b0b0b"><tspan font-weight="600">{s}</tspan> <tspan fill="#52514e">{what}, {mean[s]:.1f}% on average</tspan></text>')
        y += 22
    svg.append("</svg>")
    (HERE / "docs" / "stage-breakdown.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"wrote docs/stage-breakdown.svg from {len(runs)} runs")


if __name__ == "__main__":
    main()
