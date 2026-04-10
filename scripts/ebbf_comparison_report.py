#!/usr/bin/env python3
"""
ebbf_comparison_report.py — Compare pre- and post-ebbf cell predictions.

Generates a self-contained HTML report with:
  - Net label gain/loss bar chart (matched cells)
  - Overall entropy box plot (pre vs post)
  - Per-cell-type entropy box plots (pre vs post, grouped by pre label)
  - Per-cell-type probability simplex spider plots (log-scaled, shared axis)

Required inputs
---------------
--pre-h5ad      Pre-ebbf predictions .h5ad
                  obs must contain: cell_id (column), predicted_type (categorical)
--pre-probs     Pre-ebbf pred_probs .npy  shape (n_pre, n_classes)
--post-h5ad     Post-ebbf predictions .h5ad
                  obs index must be cell IDs matching pre cell_id values
--post-probs    Post-ebbf pred_probs .npy  shape (n_post, n_classes)
--scrna-h5ad    scRNA reference .h5ad used for class label ordering
                  obs must contain: cell_type (categorical, len == n_classes)
--output        Output HTML path  [default: ebbf_comparison.html]

Example
-------
python ebbf_comparison_report.py \\
    --pre-h5ad   predictions/sample_predictions.h5ad \\
    --pre-probs  predictions/sample_pred_probs.npy \\
    --post-h5ad  post_ebbf/predictions/sample_predictions.h5ad \\
    --post-probs post_ebbf/predictions/sample_pred_probs.npy \\
    --scrna-h5ad preprocess/sample_scrna_filtered.h5ad \\
    --output     ebbf_comparison.html
"""

import argparse
import json
import math

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import entropy as scipy_entropy

# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

EPS = 1e-4


def log_transform(arr: list[float]) -> list[float]:
    """log(p + EPS) - log(EPS): maps p=0 → 0, grows for higher p."""
    log_eps = math.log(EPS)
    return [float(math.log(max(v, 0.0) + EPS) - log_eps) for v in arr]


def back_transform(t: float) -> float:
    """Inverse of log_transform."""
    return math.exp(t + math.log(EPS)) - EPS


def top_k_indices(pre: list[float], post: list[float], k: int) -> list[int]:
    """Indices of the k classes with highest combined pre+post mean probability."""
    combined = np.array(pre) + np.array(post)
    idx = np.argsort(combined)[::-1][:k]
    return sorted(idx.tolist())


def make_global_axis(radars: list[dict], n_ticks: int = 5) -> tuple[float, list, list]:
    """Compute a shared axis max and tick positions/labels across all radars."""
    global_max = max(max(max(r["pre"]), max(r["post"])) for r in radars)
    # Cap so that no tick label back-transforms to a probability > 1.0
    max_transformed = math.log(1.0 + EPS) - math.log(EPS)
    axis_max = min(global_max * 1.05, max_transformed)
    tick_vals = [axis_max * i / n_ticks for i in range(1, n_ticks + 1)]
    tick_text = [f"p={back_transform(v):.3f}" for v in tick_vals]
    return axis_max, tick_vals, tick_text


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(pre_h5ad, pre_probs_path, post_h5ad, post_probs_path, scrna_h5ad=None):
    pre = sc.read_h5ad(pre_h5ad)
    post = sc.read_h5ad(post_h5ad)
    pre_probs = np.load(pre_probs_path)
    post_probs = np.load(post_probs_path)

    if scrna_h5ad is not None:
        scrna = sc.read_h5ad(scrna_h5ad)
        # Try major_cell_types first (6-class), fall back to cell_type
        col = "major_cell_types" if "major_cell_types" in scrna.obs.columns else "cell_type"
        class_labels = [str(c) for c in scrna.obs[col].astype("category").cat.categories]
        assert len(class_labels) == pre_probs.shape[1], (
            f"scRNA {col} categories ({len(class_labels)}) don't match "
            f"pre_probs columns ({pre_probs.shape[1]})"
        )
    else:
        # Derive class label order directly from the pre predictions
        class_labels = [str(c) for c in pre.obs["predicted_type"].astype("category").cat.categories]

    # Index pre by cell_id
    pre.obs = pre.obs.set_index("cell_id")

    pre_df = pd.DataFrame(
        {
            "label_pre": pre.obs["predicted_type"].astype(str),
            "entropy_pre": scipy_entropy(pre_probs.T),
        },
        index=pre.obs.index,
    )
    post_df = pd.DataFrame(
        {
            "label_post": post.obs["predicted_type"].astype(str),
            "entropy_post": scipy_entropy(post_probs.T),
        },
        index=post.obs.index,
    )

    pre_prob_df = pd.DataFrame(pre_probs, index=pre.obs.index, columns=class_labels)
    post_prob_df = pd.DataFrame(post_probs, index=post.obs.index, columns=class_labels)

    common = pre_df.index.intersection(post_df.index)
    only_pre = pre_df.index.difference(post_df.index)
    only_post = post_df.index.difference(pre_df.index)

    m = pre_df.loc[common].join(post_df.loc[common])

    return dict(
        class_labels=class_labels,
        pre_df=pre_df,
        post_df=post_df,
        pre_prob_df=pre_prob_df,
        post_prob_df=post_prob_df,
        matched=m,
        common=common,
        only_pre=only_pre,
        only_post=only_post,
    )


# ---------------------------------------------------------------------------
# Derived datasets
# ---------------------------------------------------------------------------


def compute_net_flow(m: pd.DataFrame) -> tuple[list, list, list]:
    changed = m[m["label_pre"] != m["label_post"]]
    lost = changed["label_pre"].value_counts()
    gained = changed["label_post"].value_counts()
    net = gained.subtract(lost, fill_value=0).astype(int).sort_values()
    labels = net.index.tolist()
    values = net.tolist()
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in values]
    return labels, values, colors


def compute_radar_data(
    m: pd.DataFrame,
    pre_prob_df: pd.DataFrame,
    post_prob_df: pd.DataFrame,
    class_labels: list[str],
    top_k: int = 20,
) -> list[dict]:
    radars = []
    for ct in sorted(m["label_pre"].unique()):
        mask = m["label_pre"] == ct
        ids = m[mask].index
        mean_pre = pre_prob_df.loc[ids].mean().tolist()
        mean_post = post_prob_df.loc[ids].mean().tolist()
        idx = top_k_indices(mean_pre, mean_post, top_k)
        radars.append(
            {
                "ct": ct,
                "n": int(mask.sum()),
                "labels": [class_labels[i] for i in idx],
                "pre": log_transform([mean_pre[i] for i in idx]),
                "post": log_transform([mean_post[i] for i in idx]),
            }
        )
    return radars


def compute_box_data(m: pd.DataFrame) -> tuple[dict, list]:
    box_data = {}
    for ct, grp in m.groupby("label_pre"):
        box_data[ct] = {
            "pre": grp["entropy_pre"].tolist(),
            "post": grp["entropy_post"].tolist(),
        }
    medians = {ct: float(np.median(v["pre"])) for ct, v in box_data.items()}
    ct_order = sorted(box_data.keys(), key=lambda ct: medians[ct])
    return box_data, ct_order


# ---------------------------------------------------------------------------
# Plotly trace builders
# ---------------------------------------------------------------------------


def bar_trace(labels, values, colors):
    return {
        "type": "bar",
        "orientation": "h",
        "x": values,
        "y": labels,
        "marker": {"color": colors},
        "hovertemplate": "<b>%{y}</b><br>Net: %{x:+d}<extra></extra>",
    }


def box_traces_paired(box_data, ct_order, overall=False):
    pre_t = dict(
        type="box",
        name="Pre-ebbf",
        marker={"color": "#4a9eff", "size": 2},
        line={"color": "#4a9eff", "width": 1.2},
        fillcolor="rgba(74,158,255,0.25)",
        boxpoints=False,
        legendgroup="pre",
        showlegend=(not overall),
        offsetgroup="pre",
        x=[],
        y=[],
        hovertemplate="<b>%{x}</b><br>Entropy: %{y:.3f}<extra>Pre</extra>",
    )
    post_t = dict(
        type="box",
        name="Post-ebbf",
        marker={"color": "#ff8c42", "size": 2},
        line={"color": "#ff8c42", "width": 1.2},
        fillcolor="rgba(255,140,66,0.25)",
        boxpoints=False,
        legendgroup="post",
        showlegend=(not overall),
        offsetgroup="post",
        x=[],
        y=[],
        hovertemplate="<b>%{x}</b><br>Entropy: %{y:.3f}<extra>Post</extra>",
    )
    for ct in ct_order:
        n = len(box_data[ct]["pre"])
        pre_t["x"].extend([ct] * n)
        pre_t["y"].extend(box_data[ct]["pre"])
        post_t["x"].extend([ct] * n)
        post_t["y"].extend(box_data[ct]["post"])
    return pre_t, post_t


def overall_box_traces(m: pd.DataFrame):
    all_pre = m["entropy_pre"].tolist()
    all_post = m["entropy_post"].tolist()
    n = len(all_pre)

    def make(vals, name, color, rgba):
        return dict(
            type="box",
            name=name,
            y=vals,
            x=[name] * n,
            marker={"color": color, "size": 2},
            line={"color": color, "width": 1.5},
            fillcolor=rgba,
            boxpoints=False,
            legendgroup=name.lower().split("-")[0],
            showlegend=False,
            offsetgroup=name,
            hovertemplate=f"Entropy: %{{y:.3f}}<extra>{name}</extra>",
        )

    return (
        make(all_pre, "Pre-ebbf", "#4a9eff", "rgba(74,158,255,0.25)"),
        make(all_post, "Post-ebbf", "#ff8c42", "rgba(255,140,66,0.25)"),
    )


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

DARK = "#1a1a1a"
BASE_LAYOUT = dict(
    paper_bgcolor=DARK,
    plot_bgcolor=DARK,
    font={"color": "#ccc"},
)


def layout_bar():
    return {
        **BASE_LAYOUT,
        "margin": {"l": 160, "r": 40, "t": 20, "b": 40},
        "xaxis": {
            "title": "Net cells gained (+) / lost (−)",
            "color": "#aaa",
            "gridcolor": "#333",
            "zeroline": True,
            "zerolinecolor": "#555",
            "zerolinewidth": 2,
        },
        "yaxis": {"color": "#ccc", "tickfont": {"size": 11}, "automargin": True},
        "bargap": 0.3,
    }


def layout_box_overall():
    return {
        **BASE_LAYOUT,
        "margin": {"l": 60, "r": 20, "t": 10, "b": 40},
        "xaxis": {"color": "#ccc", "gridcolor": "#2a2a2a"},
        "yaxis": {"title": "Entropy", "color": "#aaa", "gridcolor": "#333"},
        "boxmode": "group",
        "legend": {"font": {"color": "#ccc"}, "bgcolor": "rgba(0,0,0,0)"},
    }


def layout_box_per_type():
    return {
        **BASE_LAYOUT,
        "margin": {"l": 50, "r": 20, "t": 10, "b": 120},
        "xaxis": {
            "color": "#ccc",
            "tickfont": {"size": 9},
            "tickangle": -45,
            "gridcolor": "#2a2a2a",
            "automargin": True,
        },
        "yaxis": {"title": "Entropy", "color": "#aaa", "gridcolor": "#333"},
        "boxmode": "group",
        "legend": {"font": {"color": "#ccc"}, "bgcolor": "rgba(0,0,0,0)"},
    }


def layout_radar(ct, n, axis_max, tick_vals, tick_text):
    return {
        **BASE_LAYOUT,
        "polar": {
            "bgcolor": DARK,
            "radialaxis": {
                "visible": True,
                "color": "#ffffff",
                "gridcolor": "#3a3a3a",
                "tickfont": {"size": 8, "color": "#ffffff"},
                "tickvals": tick_vals,
                "ticktext": tick_text,
                "range": [0, axis_max],
            },
            "angularaxis": {"color": "#aaa", "tickfont": {"size": 8}, "gridcolor": "#2a2a2a"},
        },
        "title": {
            "text": f"<b>{ct}</b><br><span style='font-size:9px;color:#888'>n={n:,}</span>",
            "font": {"size": 11, "color": "#fff"},
            "x": 0.5,
            "y": 0.97,
            "xanchor": "center",
        },
        "margin": {"l": 28, "r": 28, "t": 52, "b": 16},
    }


def build_html(
    bar_labels,
    bar_values,
    bar_colors,
    overall_pre_trace,
    overall_post_trace,
    pre_box_trace,
    post_box_trace,
    radars,
    axis_max,
    tick_vals,
    tick_text,
    top_k,
    n_matched,
    n_removed,
    n_new,
):
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Pre vs Post ebbf Resegmentation</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: sans-serif; background: #0f0f0f; color: #e0e0e0; margin: 0; padding: 16px; }}
  h1 {{ font-size: 18px; font-weight: 600; color: #fff; margin-bottom: 4px; }}
  h2 {{ font-size: 14px; font-weight: 500; color: #aaa; margin: 28px 0 8px; }}
  #barplot {{ width: 100%; height: 600px; }}
  #overall-boxplot {{ width: 100%; height: 260px; }}
  #boxplot {{ width: 100%; height: 520px; }}
  #radar-container {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
    width: 100%;
  }}
  .radar-cell {{ height: 480px; width: 100%; }}
</style>
</head>
<body>
<h1>Pre vs Post ebbf Resegmentation — Cell Type Analysis</h1>
<p style="color:#888;font-size:12px">
  Matched: {n_matched:,} &nbsp;|&nbsp;
  Removed by ebbf: {n_removed:,} &nbsp;|&nbsp;
  New from ebbf: {n_new:,}
</p>

<h2>Net Label Gain / Loss per Cell Type (matched cells, label changed)</h2>
<div id="barplot"></div>

<h2>Overall Entropy — Pre vs Post ebbf (all matched cells)</h2>
<div id="overall-boxplot"></div>

<h2>Entropy per Cell Type — Pre vs Post ebbf (sorted by pre median)</h2>
<div id="boxplot"></div>

<h2>Average Probability Simplex
  <span style="font-size:11px;color:#666">
    scale: log(p + ε) − log(ε) &nbsp;|&nbsp; top-{top_k} classes per type &nbsp;|&nbsp; shared axis
  </span>
</h2>
<div style="font-size:12px;margin-bottom:10px">
  <span style="display:inline-block;width:28px;height:2px;background:#4a9eff;vertical-align:middle;margin-right:6px"></span>Pre-ebbf &nbsp;&nbsp;
  <span style="display:inline-block;width:28px;height:2px;background:#ff8c42;vertical-align:middle;margin-right:6px"></span>Post-ebbf
</div>
<div id="radar-container">
"""
    for i in range(len(radars)):
        html += f'  <div class="radar-cell" id="radar-{i}"></div>\n'

    html += "</div>\n\n<script>\n"

    # Bar chart
    html += f"Plotly.newPlot('barplot',{json.dumps([bar_trace(bar_labels, bar_values, bar_colors)])},{json.dumps(layout_bar())},{{responsive:true}});\n"

    # Overall box
    html += f"Plotly.newPlot('overall-boxplot',{json.dumps([overall_pre_trace, overall_post_trace])},{json.dumps(layout_box_overall())},{{responsive:true}});\n"

    # Per-type box
    html += f"Plotly.newPlot('boxplot',{json.dumps([pre_box_trace, post_box_trace])},{json.dumps(layout_box_per_type())},{{responsive:true}});\n"

    # Shared radar axis constants
    html += f"var TICK_VALS={json.dumps(tick_vals)};var TICK_TEXT={json.dumps(tick_text)};var AXIS_MAX={axis_max};\n"

    # Radar plots
    for i, r in enumerate(radars):
        labels = r["labels"] + [r["labels"][0]]
        pre_v = r["pre"] + [r["pre"][0]]
        post_v = r["post"] + [r["post"][0]]
        traces = [
            dict(
                type="scatterpolar",
                r=pre_v,
                theta=labels,
                fill="toself",
                line={"color": "#4a9eff", "width": 1.5},
                fillcolor="rgba(74,158,255,0.12)",
                hovertemplate="<b>%{theta}</b><br>%{r:.3f}<extra></extra>",
                showlegend=False,
            ),
            dict(
                type="scatterpolar",
                r=post_v,
                theta=labels,
                fill="toself",
                line={"color": "#ff8c42", "width": 1.5},
                fillcolor="rgba(255,140,66,0.12)",
                hovertemplate="<b>%{theta}</b><br>%{r:.3f}<extra></extra>",
                showlegend=False,
            ),
        ]
        rl = layout_radar(r["ct"], r["n"], axis_max, tick_vals, tick_text)
        # Use JS variables for shared axis so we don't repeat them 77x
        rl["polar"]["radialaxis"]["tickvals"] = "__TICK_VALS__"
        rl["polar"]["radialaxis"]["ticktext"] = "__TICK_TEXT__"
        rl["polar"]["radialaxis"]["range"] = "__AXIS_RANGE__"
        layout_str = (
            json.dumps(rl)
            .replace('"__TICK_VALS__"', "TICK_VALS")
            .replace('"__TICK_TEXT__"', "TICK_TEXT")
            .replace('"__AXIS_RANGE__"', "[0,AXIS_MAX]")
        )
        html += f"Plotly.newPlot('radar-{i}',{json.dumps(traces)},{layout_str},{{displayModeBar:false}});\n"

    html += "</script>\n</body>\n</html>"
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pre-h5ad", required=True, help="Pre-ebbf predictions .h5ad")
    parser.add_argument("--pre-probs", required=True, help="Pre-ebbf pred_probs .npy")
    parser.add_argument("--post-h5ad", required=True, help="Post-ebbf predictions .h5ad")
    parser.add_argument("--post-probs", required=True, help="Post-ebbf pred_probs .npy")
    parser.add_argument(
        "--scrna-h5ad",
        default=None,
        help="scRNA reference .h5ad (optional; obs['major_cell_types'] or 'cell_type' gives class label order; if omitted, labels are derived from pre predictions)",
    )
    parser.add_argument(
        "--output",
        default="ebbf_comparison.html",
        help="Output HTML path (default: ebbf_comparison.html)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="Top-K classes to show per radar plot (default: 20)"
    )
    args = parser.parse_args()

    print("Loading data...")
    d = load_data(args.pre_h5ad, args.pre_probs, args.post_h5ad, args.post_probs, args.scrna_h5ad)

    print(
        f"  Matched: {len(d['common']):,}  |  Removed: {len(d['only_pre']):,}  |  New: {len(d['only_post']):,}"
    )

    print("Computing net flow...")
    bar_labels, bar_values, bar_colors = compute_net_flow(d["matched"])

    print("Computing radar data...")
    radars = compute_radar_data(
        d["matched"], d["pre_prob_df"], d["post_prob_df"], d["class_labels"], args.top_k
    )
    axis_max, tick_vals, tick_text = make_global_axis(radars)

    print("Computing box plot data...")
    box_data, ct_order = compute_box_data(d["matched"])
    pre_box, post_box = box_traces_paired(box_data, ct_order)
    ov_pre, ov_post = overall_box_traces(d["matched"])

    print("Building HTML...")
    html = build_html(
        bar_labels,
        bar_values,
        bar_colors,
        ov_pre,
        ov_post,
        pre_box,
        post_box,
        radars,
        axis_max,
        tick_vals,
        tick_text,
        top_k=args.top_k,
        n_matched=len(d["common"]),
        n_removed=len(d["only_pre"]),
        n_new=len(d["only_post"]),
    )

    with open(args.output, "w") as f:
        f.write(html)
    print(f"Written to {args.output}  ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
