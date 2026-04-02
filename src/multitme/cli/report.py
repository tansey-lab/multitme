"""CLI: generate PDF reports and interactive HTML choropleth from inference results."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.backends.backend_pdf import PdfPages

from multitme.utils import configure_logging

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

TAB20 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Generate reports from inference results")
    parser.add_argument("--input", type=str, required=True, help="Path to predictions h5ad")
    parser.add_argument("--probs", type=str, required=True, help="Path to pred_probs.npy")
    parser.add_argument("--latent", type=str, required=True, help="Path to latent.npy")
    parser.add_argument(
        "--output-dir", type=str, default="results/reports", help="Output directory"
    )
    args = parser.parse_args(argv)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    logger.info("Loading data...")
    adata = sc.read_h5ad(args.input)
    probs = np.load(args.probs)
    np.load(args.latent)  # validate file exists/readable

    pred_types = adata.obs["predicted_type"].values.astype(str)
    coords = adata.obsm["spatial"]
    max_probs = probs.max(axis=1)
    n_cells = len(pred_types)

    all_types = sorted(set(pred_types))
    type_colors = {t: TAB20[i % len(TAB20)] for i, t in enumerate(all_types)}

    # Entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    max_entropy = np.log(probs.shape[1])
    norm_entropy = entropy / max_entropy

    # ── Cell type summary PDF ────────────────────────────────────────────
    logger.info("Generating cell type summary PDF...")
    _generate_summary_pdf(outdir, pred_types, all_types, type_colors, n_cells)

    # ── Confidence analysis PDF ──────────────────────────────────────────
    logger.info("Generating confidence analysis PDF...")
    _generate_confidence_pdf(
        outdir, pred_types, all_types, type_colors, max_probs, probs, norm_entropy
    )

    # ── Spatial plots PDF ────────────────────────────────────────────────
    logger.info("Generating spatial PDF...")
    _generate_spatial_pdf(
        outdir, pred_types, all_types, type_colors, coords, max_probs, norm_entropy, n_cells
    )

    # ── Interactive choropleth ───────────────────────────────────────────
    logger.info("Generating interactive choropleth...")
    _generate_choropleth(
        outdir, pred_types, all_types, type_colors, coords, max_probs, norm_entropy, n_cells
    )

    logger.info(f"Reports saved to {outdir}")


def _generate_summary_pdf(outdir, pred_types, all_types, type_colors, n_cells):
    with PdfPages(outdir / "cell_type_summary.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart
        ax = axes[0]
        counts = {t: (pred_types == t).sum() for t in all_types}
        sorted_types = sorted(counts, key=counts.get, reverse=True)
        bars = ax.barh(
            range(len(sorted_types)),
            [counts[t] for t in sorted_types],
            color=[type_colors[t] for t in sorted_types],
        )
        ax.set_yticks(range(len(sorted_types)))
        ax.set_yticklabels(sorted_types, fontsize=10)
        ax.set_xlabel("Number of cells")
        ax.set_title("Predicted Cell Type Distribution", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        for bar, t in zip(bars, sorted_types, strict=False):
            pct = 100 * counts[t] / n_cells
            ax.text(
                bar.get_width() + max(counts[sorted_types[0]] * 0.01, 50),
                bar.get_y() + bar.get_height() / 2,
                f"{counts[t]:,} ({pct:.1f}%)",
                va="center",
                fontsize=9,
            )

        # Pie chart
        ax = axes[1]
        top_n = 8
        top_types = sorted_types[:top_n]
        top_counts = [counts[t] for t in top_types]
        other_count = sum(counts[t] for t in sorted_types[top_n:])
        if other_count > 0:
            labels_pie = top_types + ["Other"]
            counts_pie = top_counts + [other_count]
            colors_pie = [type_colors[t] for t in top_types] + ["#888888"]
        else:
            labels_pie = top_types
            counts_pie = top_counts
            colors_pie = [type_colors[t] for t in top_types]
        ax.pie(
            counts_pie,
            labels=labels_pie,
            colors=colors_pie,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 9},
        )
        ax.set_title("Composition", fontsize=13, fontweight="bold")

        fig.suptitle(
            f"MultiTME Cell Type Predictions ({n_cells:,} cells)",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _generate_confidence_pdf(
    outdir, pred_types, all_types, type_colors, max_probs, probs, norm_entropy
):
    with PdfPages(outdir / "confidence_analysis.pdf") as pdf:
        # Page 1: Overall confidence distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.hist(max_probs, bins=50, color="#4a90d9", edgecolor="white", linewidth=0.3)
        ax.axvline(
            x=np.median(max_probs),
            color="red",
            linestyle="--",
            label=f"Median: {np.median(max_probs):.3f}",
        )
        ax.axvline(x=0.5, color="orange", linestyle="--", alpha=0.7, label="0.5 threshold")
        ax.set_xlabel("Max prediction probability")
        ax.set_ylabel("Number of cells")
        ax.set_title("Confidence Distribution", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)

        ax = axes[1]
        thresholds = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        pcts = [100 * (max_probs >= t).sum() / len(max_probs) for t in thresholds]
        ax.bar(range(len(thresholds)), pcts, color="#4a90d9", edgecolor="white")
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f"\u2265{t}" for t in thresholds])
        ax.set_ylabel("% of cells")
        ax.set_title("Cells Above Confidence Threshold", fontsize=13, fontweight="bold")
        for i, (_t, p) in enumerate(zip(thresholds, pcts, strict=False)):
            ax.text(i, p + 1, f"{p:.1f}%", ha="center", fontsize=9)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Per-type confidence boxplots
        fig, ax = plt.subplots(figsize=(12, 6))
        box_data = []
        box_labels = []
        sorted_by_count = sorted(all_types, key=lambda t: (pred_types == t).sum(), reverse=True)
        for t in sorted_by_count:
            mask = pred_types == t
            if mask.sum() > 0:
                box_data.append(max_probs[mask])
                box_labels.append(f"{t}\n(n={mask.sum():,})")

        bp = ax.boxplot(box_data, vert=True, patch_artist=True, showfliers=False)
        for patch, t in zip(bp["boxes"], sorted_by_count[: len(box_data)], strict=False):
            patch.set_facecolor(type_colors[t])
            patch.set_alpha(0.7)
        ax.set_xticklabels(box_labels, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel("Prediction confidence")
        ax.set_title("Confidence by Predicted Cell Type", fontsize=13, fontweight="bold")
        ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: Entropy distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(norm_entropy, bins=50, color="#e07b39", edgecolor="white", linewidth=0.3)
        ax.axvline(
            x=np.median(norm_entropy),
            color="red",
            linestyle="--",
            label=f"Median: {np.median(norm_entropy):.3f}",
        )
        ax.set_xlabel("Normalized entropy (0=certain, 1=uniform)")
        ax.set_ylabel("Number of cells")
        ax.set_title("Prediction Entropy Distribution", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _generate_spatial_pdf(
    outdir, pred_types, all_types, type_colors, coords, max_probs, norm_entropy, n_cells
):
    with PdfPages(outdir / "spatial_plots.pdf") as pdf:
        if n_cells > 80000:
            idx = np.random.RandomState(42).choice(n_cells, 80000, replace=False)
        else:
            idx = np.arange(n_cells)

        xy = coords[idx]
        pt = pred_types[idx]
        mp = max_probs[idx]
        ne = norm_entropy[idx]

        # Page 1: Spatial predicted types
        fig, ax = plt.subplots(figsize=(14, 12))
        for t in all_types:
            mask = pt == t
            if mask.sum() == 0:
                continue
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=0.3,
                c=type_colors[t],
                label=f"{t} ({mask.sum():,})",
                alpha=0.5,
                rasterized=True,
            )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title("Predicted Cell Types (spatial)", fontsize=14, fontweight="bold")
        ax.legend(
            markerscale=10, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False
        )
        ax.set_xlabel("X (\u03bcm)")
        ax.set_ylabel("Y (\u03bcm)")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

        # Page 2: Spatial confidence
        fig, ax = plt.subplots(figsize=(14, 12))
        order = np.argsort(mp)
        sc_plot = ax.scatter(
            xy[order, 0],
            xy[order, 1],
            s=0.3,
            c=mp[order],
            cmap="RdYlGn",
            vmin=0.3,
            vmax=1.0,
            alpha=0.6,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title("Prediction Confidence (spatial)", fontsize=14, fontweight="bold")
        plt.colorbar(sc_plot, ax=ax, shrink=0.6, label="Max probability")
        ax.set_xlabel("X (\u03bcm)")
        ax.set_ylabel("Y (\u03bcm)")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

        # Page 3: Spatial entropy
        fig, ax = plt.subplots(figsize=(14, 12))
        order = np.argsort(-ne)
        sc_plot = ax.scatter(
            xy[order, 0],
            xy[order, 1],
            s=0.3,
            c=ne[order],
            cmap="YlOrRd",
            vmin=0,
            vmax=0.5,
            alpha=0.6,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title("Prediction Entropy (spatial)", fontsize=14, fontweight="bold")
        plt.colorbar(sc_plot, ax=ax, shrink=0.6, label="Normalized entropy")
        ax.set_xlabel("X (\u03bcm)")
        ax.set_ylabel("Y (\u03bcm)")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

        # Page 4: Per-type spatial panels
        counts = {t: (pt == t).sum() for t in all_types}
        top_types_spatial = sorted(counts, key=counts.get, reverse=True)[:12]
        n_panels = len(top_types_spatial)
        ncols = 4
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
        axes = axes.ravel()
        for i, t in enumerate(top_types_spatial):
            ax = axes[i]
            mask = pt == t
            ax.scatter(xy[:, 0], xy[:, 1], s=0.1, c="#333333", alpha=0.1, rasterized=True)
            if mask.sum() > 0:
                ax.scatter(
                    xy[mask, 0], xy[mask, 1], s=0.4, c=type_colors[t], alpha=0.6, rasterized=True
                )
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_title(f"{t} (n={mask.sum():,})", fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
        for i in range(n_panels, len(axes)):
            axes[i].set_visible(False)
        fig.suptitle("Per-Type Spatial Distribution", fontsize=15, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)


def _generate_choropleth(
    outdir, pred_types, all_types, type_colors, coords, max_probs, norm_entropy, n_cells
):
    max_interactive = 100000
    if n_cells > max_interactive:
        idx = np.random.RandomState(42).choice(n_cells, max_interactive, replace=False)
    else:
        idx = np.arange(n_cells)

    xy_i = coords[idx]
    pt_i = pred_types[idx].tolist()
    mp_i = max_probs[idx].tolist()
    ne_i = norm_entropy[idx].tolist()

    xy_list = [[round(float(x), 1), round(float(y), 1)] for x, y in xy_i]
    xmin, xmax = float(xy_i[:, 0].min()), float(xy_i[:, 0].max())
    ymin, ymax = float(xy_i[:, 1].min()), float(xy_i[:, 1].max())

    type_stats = {}
    for t in all_types:
        mask = np.array(pt_i) == t
        n = int(mask.sum())
        if n > 0:
            confs = np.array(mp_i)[mask]
            type_stats[t] = {
                "n": n,
                "mean_conf": round(float(confs.mean()), 3),
                "med_conf": round(float(np.median(confs)), 3),
            }
        else:
            type_stats[t] = {"n": 0, "mean_conf": 0, "med_conf": 0}

    html = _CHOROPLETH_TEMPLATE.format(
        coords_json=json.dumps(xy_list),
        pred_json=json.dumps(pt_i),
        conf_json=json.dumps([round(p, 3) for p in mp_i]),
        entropy_json=json.dumps([round(e, 3) for e in ne_i]),
        types_json=json.dumps(all_types),
        colors_json=json.dumps(type_colors),
        stats_json=json.dumps(type_stats),
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        n_shown=len(idx),
        n_total=n_cells,
    )

    html_path = outdir / "choropleth.html"
    html_path.write_text(html)
    fsize = html_path.stat().st_size / 1e6
    logger.info(f"Saved {html_path} ({fsize:.1f} MB)")


_CHOROPLETH_TEMPLATE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>MultiTME Cell Type Choropleth</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         background: #1a1a2e; color: #eee; overflow: hidden; }}
  #controls {{ position: fixed; top: 0; left: 0; right: 0; z-index: 10;
               background: rgba(26,26,46,0.95); padding: 10px 20px;
               display: flex; align-items: center; gap: 15px; flex-wrap: wrap;
               border-bottom: 1px solid #333; }}
  #controls label {{ font-size: 13px; color: #aaa; }}
  #controls input {{ font-size: 13px; padding: 4px 8px;
               background: #2a2a4a; color: #eee; border: 1px solid #444; border-radius: 4px; }}
  .btn {{ padding: 5px 12px; background: #3a3a6a; color: #eee; border: 1px solid #555;
          border-radius: 4px; cursor: pointer; font-size: 12px; }}
  .btn.active {{ background: #5a5aaa; border-color: #88f; }}
  .btn:hover {{ background: #4a4a8a; }}
  #stats {{ font-size: 12px; color: #ccc; margin-left: auto; }}
  canvas {{ display: block; cursor: grab; }}
  canvas:active {{ cursor: grabbing; }}
  #tooltip {{ position: fixed; pointer-events: none; background: rgba(0,0,0,0.9);
              color: #fff; padding: 8px 12px; border-radius: 4px; font-size: 12px;
              display: none; z-index: 20; border: 1px solid #555; max-width: 300px; }}
  #legend {{ position: fixed; bottom: 20px; right: 20px; background: rgba(26,26,46,0.9);
             padding: 12px; border-radius: 6px; border: 1px solid #444; z-index: 10;
             font-size: 11px; max-height: 80vh; overflow-y: auto; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; cursor: pointer; }}
  .legend-item:hover {{ opacity: 0.8; }}
  .legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }}
  #legend canvas {{ border-radius: 3px; }}
  #legend-labels {{ display: flex; justify-content: space-between; font-size: 11px;
                    color: #aaa; margin-top: 4px; }}
</style>
</head><body>

<div id="controls">
  <button class="btn active" id="btn-pred" onclick="setMode('pred')">Cell Type</button>
  <button class="btn" id="btn-conf" onclick="setMode('conf')">Confidence</button>
  <button class="btn" id="btn-entropy" onclick="setMode('entropy')">Entropy</button>
  <label>Point size:
    <input type="range" id="size-slider" min="1" max="10" value="3" style="width:80px;">
  </label>
  <label>Min confidence:
    <input type="range" id="conf-thresh" min="0" max="100" value="0" style="width:80px;">
    <span id="conf-thresh-label">0%</span>
  </label>
  <span style="font-size:12px;color:#666;">Scroll=zoom, drag=pan</span>
  <span id="stats">{n_shown:,} cells shown (of {n_total:,})</span>
</div>

<canvas id="map"></canvas>
<div id="tooltip"></div>
<div id="legend"></div>

<script>
const COORDS = {coords_json};
const PRED = {pred_json};
const CONF = {conf_json};
const ENTROPY = {entropy_json};
const TYPES = {types_json};
const COLORS = {colors_json};
const TYPE_STATS = {stats_json};
const BOUNDS = {{ xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax} }};
const N = COORDS.length;

let mode = "pred";
let transform = {{ x: 0, y: 0, scale: 1 }};
let isDragging = false, dragStart = {{ x: 0, y: 0 }};
let highlightType = null;
let confThreshold = 0;
let pointSize = 3;

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const legendEl = document.getElementById("legend");

function hexToRgb(hex) {{
  return [parseInt(hex.slice(1,3),16), parseInt(hex.slice(3,5),16), parseInt(hex.slice(5,7),16)];
}}

function confToColor(c) {{
  if (c < 0.5) {{ const t = c/0.5; return [220, Math.round(60+t*180), Math.round(30+t*20)]; }}
  else {{ const t = (c-0.5)/0.5; return [Math.round(220-t*180), Math.round(240-t*40), Math.round(50-t*20)]; }}
}}

function entropyToColor(e) {{
  const t = Math.min(e / 0.5, 1);
  return [Math.round(50+t*200), Math.round(100+(1-t)*100), Math.round(220-t*180)];
}}

function setMode(m) {{
  mode = m;
  document.getElementById("btn-pred").classList.toggle("active", m === "pred");
  document.getElementById("btn-conf").classList.toggle("active", m === "conf");
  document.getElementById("btn-entropy").classList.toggle("active", m === "entropy");
  highlightType = null;
  updateLegend();
  render();
}}

function updateLegend() {{
  legendEl.innerHTML = "";
  if (mode === "pred") {{
    const counts = {{}};
    TYPES.forEach(t => counts[t] = 0);
    for (let i = 0; i < N; i++) {{ if (CONF[i] >= confThreshold) counts[PRED[i]]++; }}
    const sorted = TYPES.slice().sort((a,b) => counts[b] - counts[a]);
    sorted.forEach(t => {{
      const div = document.createElement("div");
      div.className = "legend-item";
      div.innerHTML = `<div class="legend-swatch" style="background:${{COLORS[t]}}"></div>` +
        `<span>${{t}} (${{counts[t].toLocaleString()}})</span>`;
      div.onclick = () => {{ highlightType = highlightType === t ? null : t; render(); }};
      legendEl.appendChild(div);
    }});
  }} else {{
    const lc = document.createElement("canvas");
    lc.width = 200; lc.height = 15;
    const lctx = lc.getContext("2d");
    for (let i = 0; i < 200; i++) {{
      const v = i / 200;
      let [r,g,b] = mode === "conf" ? confToColor(v) : entropyToColor(v * 0.5);
      lctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
      lctx.fillRect(i, 0, 1, 15);
    }}
    legendEl.appendChild(lc);
    const labels = document.createElement("div");
    labels.id = "legend-labels";
    labels.innerHTML = mode === "conf"
      ? `<span>Low</span><span>0.5</span><span>High</span>`
      : `<span>Certain</span><span>Uncertain</span>`;
    legendEl.appendChild(labels);
  }}
}}

function resize() {{
  canvas.width = window.innerWidth; canvas.height = window.innerHeight;
  const w = canvas.width, h = canvas.height - 50;
  const dW = BOUNDS.xmax - BOUNDS.xmin, dH = BOUNDS.ymax - BOUNDS.ymin;
  const s = Math.min(w / dW, h / dH) * 0.9;
  transform.scale = s;
  transform.x = (w - dW * s) / 2 - BOUNDS.xmin * s;
  transform.y = (h - dH * s) / 2 - BOUNDS.ymin * s + 50;
  render();
}}

function render() {{
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const s = transform.scale, tx = transform.x, ty = transform.y;
  const r = pointSize * Math.max(0.5, Math.min(s / 2, 3));

  for (let i = 0; i < N; i++) {{
    if (CONF[i] < confThreshold) continue;
    const sx = COORDS[i][0]*s+tx, sy = COORDS[i][1]*s+ty;
    if (sx < -r || sx > w+r || sy < -r || sy > h+r) continue;

    let rgb, alpha;
    if (mode === "conf") {{ rgb = confToColor(CONF[i]); alpha = 0.7; }}
    else if (mode === "entropy") {{ rgb = entropyToColor(ENTROPY[i]); alpha = 0.7; }}
    else {{
      const t = PRED[i];
      alpha = (highlightType && highlightType !== t) ? 0.05 : 0.6;
      rgb = hexToRgb(COLORS[t]);
    }}
    ctx.fillStyle = `rgba(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}},${{alpha}})`;
    ctx.fillRect(sx-r/2, sy-r/2, r, r);
  }}
}}

canvas.addEventListener("wheel", e => {{
  e.preventDefault();
  const f = e.deltaY > 0 ? 0.85 : 1.18;
  transform.x = e.clientX - (e.clientX - transform.x) * f;
  transform.y = e.clientY - (e.clientY - transform.y) * f;
  transform.scale *= f;
  render();
}});
canvas.addEventListener("mousedown", e => {{
  isDragging = true;
  dragStart.x = e.clientX - transform.x; dragStart.y = e.clientY - transform.y;
}});
canvas.addEventListener("mousemove", e => {{
  if (isDragging) {{
    transform.x = e.clientX - dragStart.x; transform.y = e.clientY - dragStart.y;
    render(); return;
  }}
  const s = transform.scale, tx = transform.x, ty = transform.y;
  const mx = (e.clientX-tx)/s, my = (e.clientY-ty)/s;
  let best = -1, bestD = 25;
  for (let i = 0; i < N; i++) {{
    const dx = COORDS[i][0]-mx, dy = COORDS[i][1]-my;
    const d = dx*dx+dy*dy;
    if (d < bestD) {{ bestD = d; best = i; }}
  }}
  if (best >= 0) {{
    tooltip.style.display = "block";
    tooltip.style.left = (e.clientX+12)+"px";
    tooltip.style.top = (e.clientY+12)+"px";
    tooltip.innerHTML =
      `<b>Cell Type:</b> ${{PRED[best]}}<br>` +
      `<b>Confidence:</b> ${{(CONF[best]*100).toFixed(1)}}%<br>` +
      `<b>Entropy:</b> ${{ENTROPY[best].toFixed(3)}}`;
  }} else {{ tooltip.style.display = "none"; }}
}});
canvas.addEventListener("mouseup", () => {{ isDragging = false; }});
canvas.addEventListener("mouseleave", () => {{ isDragging = false; tooltip.style.display = "none"; }});

document.getElementById("size-slider").addEventListener("input", e => {{
  pointSize = parseInt(e.target.value); render();
}});
document.getElementById("conf-thresh").addEventListener("input", e => {{
  confThreshold = parseInt(e.target.value) / 100;
  document.getElementById("conf-thresh-label").textContent = e.target.value + "%";
  updateLegend(); render();
}});

window.addEventListener("resize", resize);
resize();
updateLegend();
</script>
</body></html>"""


if __name__ == "__main__":
    main()
