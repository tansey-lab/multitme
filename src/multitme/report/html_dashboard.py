"""Single-file HTML dashboard (Plotly + embedded figures) for MultiTME."""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scanpy as sc
from plotly.subplots import make_subplots

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def compute_gene_overlap_stats(scrna_path: Path, xenium_path: Path) -> dict[str, int | list[str]]:
    """Return counts and sorted list of shared gene symbols."""
    s_adata = sc.read_h5ad(scrna_path)
    x_adata = sc.read_h5ad(xenium_path)
    genes_s = set(s_adata.var_names.astype(str))
    genes_x = set(x_adata.var_names.astype(str))
    common = genes_s & genes_x
    only_s = genes_s - genes_x
    only_x = genes_x - genes_s
    return {
        "n_scrna_only": len(only_s),
        "n_xenium_only": len(only_x),
        "n_common": len(common),
        "n_scrna_total": len(genes_s),
        "n_xenium_total": len(genes_x),
        "common_genes": sorted(common),
    }


def write_html_dashboard(
    outdir: Path,
    *,
    sample_prefix: str,
    celltype_counts_path: Path | None,
    gene_overlap: dict[str, int | list[str]] | None,
    pred_types: np.ndarray,
    all_types: list[str],
    type_colors: dict[str, str],
    coords: np.ndarray,
    max_probs: np.ndarray,
    norm_entropy: np.ndarray,
    n_cells: int,
    total_transcripts_per_cell: np.ndarray | None = None,
) -> None:
    """Write ``report.html`` with interactive Plotly figures and asset links."""
    sections: list[str] = []

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>'

    css = """
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
           margin: 0; background: #f6f7f9; color: #1a1a1a; }
    main { max-width: 1200px; margin: 0 auto; padding: 24px 20px 48px; }
    h1 { font-size: 1.5rem; margin: 0 0 8px; }
    h2 { font-size: 1.15rem; margin: 32px 0 12px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
    p.sub { color: #555; margin: 0 0 24px; font-size: 0.95rem; }
    .card { background: #fff; border-radius: 8px; padding: 16px; margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,.08); }
    .links a { margin-right: 16px; }
    .muted { color: #666; font-size: 0.9rem; }
    iframe { width: 100%; height: 720px; border: 1px solid #ddd; border-radius: 6px; background: #fff; }
    table.genes { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
    table.genes th, table.genes td { border: 1px solid #e0e0e0; padding: 8px 10px; text-align: left; }
    table.genes th { background: #f0f0f0; }
    nav { background: #fff; border-bottom: 1px solid #e0e0e0; padding: 12px 20px; position: sticky; top: 0; z-index: 10; }
    nav a { margin-right: 14px; color: #1a5f9e; text-decoration: none; font-size: 0.9rem; }
    nav a:hover { text-decoration: underline; }
    """

    nav = """
    <nav>
      <a href="#input-scrna">Input scRNA</a>
      <a href="#genes">Gene overlap</a>
      <a href="#predictions">Predictions</a>
      <a href="#transcripts">Transcripts vs type</a>
      <a href="#confidence">Confidence</a>
      <a href="#spatial">Spatial</a>
      <a href="#assets">PDFs & choropleth</a>
    </nav>
    """

    def plotly_block(fig: go.Figure) -> str:
        block = fig.to_html(
            include_plotlyjs=False,
            full_html=False,
            config={"displayModeBar": True, "responsive": True},
        )
        return f'<div class="card">{block}</div>'

    # ── Input scRNA annotations ─────────────────────────────────────────
    input_section = _section_input_scrna(celltype_counts_path)
    if input_section:
        sections.append(f'<h2 id="input-scrna">Input scRNA annotations</h2>{input_section}')

    # ── Gene overlap ────────────────────────────────────────────────────
    gene_section, gene_table_html = _section_gene_overlap(gene_overlap)
    sections.append(f'<h2 id="genes">Gene overlap (scRNA vs Xenium)</h2>{gene_section}')
    if gene_table_html:
        sections.append(f'<div class="card">{gene_table_html}</div>')

    # ── Predicted distribution ─────────────────────────────────────────
    fig_pred = _fig_predicted_distribution(pred_types, all_types, type_colors, n_cells)
    sections.append('<h2 id="predictions">Xenium predictions</h2>' + plotly_block(fig_pred))

    if total_transcripts_per_cell is not None and len(total_transcripts_per_cell) == n_cells:
        fig_tx = _fig_transcripts_vs_classification(
            pred_types, all_types, type_colors, total_transcripts_per_cell
        )
        tx_table = _html_transcript_summary_table(pred_types, all_types, total_transcripts_per_cell)
        sections.append(
            '<h2 id="transcripts">Total transcripts per cell vs predicted type</h2>'
            + '<p class="muted">Values are the sum of the expression matrix per cell in the '
            "predictions object (Xenium layer; counts or normalized counts depending on input).</p>"
            + plotly_block(fig_tx)
            + f'<div class="card">{tx_table}</div>'
        )

    # ── Confidence ─────────────────────────────────────────────────────
    fig_conf = _fig_confidence(max_probs, norm_entropy, pred_types, all_types, type_colors)
    sections.append('<h2 id="confidence">Prediction confidence</h2>' + plotly_block(fig_conf))

    # ── Spatial ────────────────────────────────────────────────────────
    fig_spatial = _fig_spatial(coords, pred_types, all_types, type_colors, max_probs, norm_entropy)
    sections.append('<h2 id="spatial">Spatial (Xenium)</h2>' + plotly_block(fig_spatial))

    # ── Links to PDFs / choropleth ─────────────────────────────────────
    assets = _section_asset_links(sample_prefix)
    sections.append('<h2 id="assets">PDF exports & interactive choropleth</h2>' + assets)

    title = f"{sample_prefix.rstrip('_')} — MultiTME report" if sample_prefix else "MultiTME report"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  {plotly_cdn}
  <style>{css}</style>
</head>
<body>
{nav}
<main>
  <h1>{title}</h1>
  <p class="sub">Interactive summary. UMAP views are exported as PDF alongside this file.</p>
  {"".join(sections)}
  <p class="muted" style="margin-top:32px">Generated by multitme-report</p>
</main>
</body>
</html>
"""

    out_path = outdir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("Wrote %s", out_path)

    genes_txt = outdir / "gene_overlap_common_genes.txt"
    if gene_overlap is not None:
        cg = gene_overlap.get("common_genes") or []
        genes_txt.write_text("\n".join(cg) + ("\n" if cg else ""), encoding="utf-8")
        logger.info("Wrote %s", genes_txt)


def _section_input_scrna(celltype_counts_path: Path | None) -> str:
    if not celltype_counts_path or not Path(celltype_counts_path).is_file():
        return (
            '<div class="card"><p class="muted">No scrna_celltype_counts.json provided; '
            "skipping input annotation breakdown.</p></div>"
        )
    data = json.loads(Path(celltype_counts_path).read_text(encoding="utf-8"))
    per = data.get("per_cell_type") or []
    if not per:
        return '<div class="card"><p class="muted">Empty cell-type table.</p></div>'

    ann = data.get("annotation_column", "annotation")
    down = bool(data.get("downsampled", False))
    total_b = data.get("total_before")
    total_a = data.get("total_after")

    types = [row["cell_type"] for row in per]
    before = [row["n_before"] for row in per]
    after = [row["n_after"] for row in per]

    if down:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Before downsampling",
                x=before,
                y=types,
                orientation="h",
                marker_color="#6baed6",
            )
        )
        fig.add_trace(
            go.Bar(
                name="After downsampling", x=after, y=types, orientation="h", marker_color="#fd8d3c"
            )
        )
        fig.update_layout(
            barmode="group",
            title=f"Cells per type ({ann}) — QC & downsampling",
            xaxis_title="Cells",
            yaxis=dict(autorange="reversed"),
            height=max(400, 28 * len(types)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=120, r=24, t=60, b=40),
        )
        fig.add_annotation(
            text=f"Total: {total_b:,} → {total_a:,} cells",
            xref="paper",
            yref="paper",
            x=0,
            y=1.08,
            showarrow=False,
            font=dict(size=12),
        )
    else:
        fig = go.Figure(
            go.Bar(
                x=after,
                y=types,
                orientation="h",
                marker_color="#6baed6",
                name="Cells",
            )
        )
        fig.update_layout(
            title=f"Cells per type ({ann}) — input scRNA used for training",
            xaxis_title="Cells",
            yaxis=dict(autorange="reversed"),
            height=max(400, 28 * len(types)),
            margin=dict(l=120, r=24, t=60, b=40),
        )

    block = fig.to_html(include_plotlyjs=False, full_html=False, config={"responsive": True})
    return f'<div class="card">{block}</div>'


def _section_gene_overlap(
    stats: dict[str, int | list[str]] | None,
) -> tuple[str, str]:
    if not stats:
        return (
            '<div class="card"><p class="muted">Gene overlap was not computed '
            "(missing scRNA/Xenium inputs).</p></div>",
            "",
        )

    n_s = int(stats["n_scrna_only"])
    n_x = int(stats["n_xenium_only"])
    n_ab = int(stats["n_common"])
    n_st = int(stats["n_scrna_total"])
    n_xt = int(stats["n_xenium_total"])

    venn_html = ""
    try:
        from matplotlib_venn import venn2

        fig, ax = plt.subplots(figsize=(7, 7))
        venn2(subsets=(n_s, n_x, n_ab), set_labels=("scRNA", "Xenium"), ax=ax)
        ax.set_title("Gene symbols overlap")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        venn_html = (
            f'<div class="card"><img src="data:image/png;base64,{b64}" '
            'alt="Venn diagram" style="max-width:100%;height:auto"/></div>'
        )
    except Exception as e:
        logger.warning("Venn diagram failed (%s); using bar fallback", e)
        fig = go.Figure(
            go.Bar(
                x=["scRNA only", "Shared", "Xenium only"],
                y=[n_s, n_ab, n_x],
                marker_color=["#6baed6", "#74c476", "#fd8d3c"],
            )
        )
        fig.update_layout(
            title="Gene overlap",
            yaxis_title="Number of genes",
            height=400,
        )
        venn_html = fig.to_html(
            include_plotlyjs=False, full_html=False, config={"responsive": True}
        )

    table = f"""
    <table class="genes">
      <tr><th>Category</th><th>Count</th></tr>
      <tr><td>scRNA only</td><td>{n_s:,}</td></tr>
      <tr><td>Shared</td><td>{n_ab:,}</td></tr>
      <tr><td>Xenium only</td><td>{n_x:,}</td></tr>
      <tr><td>scRNA total</td><td>{n_st:,}</td></tr>
      <tr><td>Xenium total</td><td>{n_xt:,}</td></tr>
    </table>
    """
    return venn_html, table


def _fig_predicted_distribution(
    pred_types: np.ndarray,
    all_types: list[str],
    type_colors: dict[str, str],
    n_cells: int,
) -> go.Figure:
    counts = {t: int((pred_types == t).sum()) for t in all_types}
    sorted_types = sorted(counts, key=counts.get, reverse=True)
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Predicted types (bar)", "Composition (pie)")
    )

    fig.add_trace(
        go.Bar(
            x=[counts[t] for t in sorted_types],
            y=sorted_types,
            orientation="h",
            marker_color=[type_colors[t] for t in sorted_types],
            name="cells",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    top_n = 8
    top_types = sorted_types[:top_n]
    top_counts = [counts[t] for t in top_types]
    other = sum(counts[t] for t in sorted_types[top_n:])
    labels = top_types + (["Other"] if other > 0 else [])
    values = top_counts + ([other] if other > 0 else [])
    colors_p = [type_colors[t] for t in top_types] + (["#888888"] if other > 0 else [])

    fig.add_trace(
        go.Pie(labels=labels, values=values, marker=dict(colors=colors_p), name=""),
        row=1,
        col=2,
    )
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_layout(
        title_text=f"Predicted cell types on Xenium ({n_cells:,} cells)",
        height=max(420, 24 * len(sorted_types)),
        margin=dict(l=100, r=24, t=80, b=40),
    )
    return fig


def _fig_confidence(
    max_probs: np.ndarray,
    norm_entropy: np.ndarray,
    pred_types: np.ndarray,
    all_types: list[str],
    type_colors: dict[str, str],
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Max probability (all cells)",
            "Cells above threshold",
            "Confidence by predicted type",
            "Normalized entropy",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    fig.add_trace(
        go.Histogram(x=max_probs, nbinsx=50, marker_color="#4a90d9", name=""),
        row=1,
        col=1,
    )
    med = float(np.median(max_probs))
    fig.add_vline(
        x=med,
        line_dash="dash",
        line_color="red",
        annotation_text=f"median {med:.3f}",
        row=1,
        col=1,
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="orange", row=1, col=1)

    thresholds = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    pcts = [100 * float((max_probs >= t).sum()) / len(max_probs) for t in thresholds]
    fig.add_trace(
        go.Bar(
            x=[f"≥{t}" for t in thresholds],
            y=pcts,
            marker_color="#4a90d9",
            name="",
            text=[f"{p:.1f}%" for p in pcts],
            textposition="outside",
        ),
        row=1,
        col=2,
    )

    sorted_by_count = sorted(all_types, key=lambda t: (pred_types == t).sum(), reverse=True)
    for t in sorted_by_count:
        mask = pred_types == t
        if mask.sum() == 0:
            continue
        fig.add_trace(
            go.Box(
                y=max_probs[mask],
                name=t,
                marker_color=type_colors[t],
                boxmean=True,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Histogram(x=norm_entropy, nbinsx=50, marker_color="#e07b39", name=""),
        row=2,
        col=2,
    )
    fig.add_vline(
        x=float(np.median(norm_entropy)),
        line_dash="dash",
        line_color="red",
        annotation_text="median",
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Max probability", row=1, col=1)
    fig.update_yaxes(title_text="Cells", row=1, col=1)
    fig.update_yaxes(title_text="% of cells", row=1, col=2)
    fig.update_xaxes(title_text="Predicted type", row=2, col=1)
    fig.update_yaxes(title_text="Max probability", row=2, col=1)
    fig.update_xaxes(title_text="Normalized entropy", row=2, col=2)
    fig.update_yaxes(title_text="Cells", row=2, col=2)

    fig.update_layout(height=780, showlegend=False, title_text="Confidence analysis")
    return fig


def _fig_spatial(
    coords: np.ndarray,
    pred_types: np.ndarray,
    all_types: list[str],
    type_colors: dict[str, str],
    max_probs: np.ndarray,
    norm_entropy: np.ndarray,
) -> go.Figure:
    n_cells = len(pred_types)
    max_pts = 80_000
    if n_cells > max_pts:
        idx = np.random.RandomState(42).choice(n_cells, max_pts, replace=False)
    else:
        idx = np.arange(n_cells)
    xy = coords[idx]
    pt = pred_types[idx]
    mp = max_probs[idx]
    ne = norm_entropy[idx]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Predicted type", "Confidence", "Entropy"),
        horizontal_spacing=0.04,
    )
    for _, t in enumerate(all_types):
        mask = pt == t
        if mask.sum() == 0:
            continue
        fig.add_trace(
            go.Scattergl(
                x=xy[mask, 0],
                y=xy[mask, 1],
                mode="markers",
                marker=dict(size=2, color=type_colors[t], opacity=0.45),
                name=t,
                legendgroup=t,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    order = np.argsort(mp)
    fig.add_trace(
        go.Scattergl(
            x=xy[order, 0],
            y=xy[order, 1],
            mode="markers",
            marker=dict(
                size=2, color=mp[order], colorscale="RdYlGn", cmin=0.3, cmax=1.0, opacity=0.55
            ),
            name="conf",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    order_e = np.argsort(-ne)
    fig.add_trace(
        go.Scattergl(
            x=xy[order_e, 0],
            y=xy[order_e, 1],
            mode="markers",
            marker=dict(
                size=2, color=ne[order_e], colorscale="YlOrRd", cmin=0, cmax=0.5, opacity=0.55
            ),
            name="entropy",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_yaxes(scaleanchor="x3", scaleratio=1, row=1, col=3)
    for c in (1, 2, 3):
        fig.update_yaxes(autorange="reversed", row=1, col=c)

    fig.update_layout(
        height=520,
        title_text=f"Spatial overview ({len(idx):,} of {n_cells:,} cells)",
        legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.02),
    )
    fig.update_xaxes(title_text="x (µm)", row=1, col=1)
    fig.update_yaxes(title_text="y (µm)", row=1, col=1)
    return fig


def _fig_transcripts_vs_classification(
    pred_types: np.ndarray,
    all_types: list[str],
    type_colors: dict[str, str],
    total_tx: np.ndarray,
) -> go.Figure:
    log_tx = np.log10(total_tx.astype(np.float64) + 1.0)
    sorted_by_count = sorted(all_types, key=lambda t: (pred_types == t).sum(), reverse=True)
    fig = go.Figure()
    n_present = 0
    for t in sorted_by_count:
        mask = pred_types == t
        if mask.sum() == 0:
            continue
        n_present += 1
        fig.add_trace(
            go.Box(
                y=log_tx[mask],
                name=t,
                marker_color=type_colors[t],
                line=dict(color="#333"),
            )
        )
    fig.update_layout(
        title="Total transcripts per cell vs predicted cell type (log10(counts + 1))",
        xaxis_title="Predicted cell type",
        yaxis_title="log10(total counts per cell + 1)",
        height=max(440, min(900, 28 * n_present)),
        showlegend=False,
        xaxis_tickangle=-40,
        margin=dict(b=120),
    )
    return fig


def _html_transcript_summary_table(
    pred_types: np.ndarray,
    all_types: list[str],
    total_tx: np.ndarray,
) -> str:
    sorted_by_count = sorted(all_types, key=lambda t: (pred_types == t).sum(), reverse=True)
    rows: list[str] = []
    for t in sorted_by_count:
        mask = pred_types == t
        n = int(mask.sum())
        if n == 0:
            continue
        v = total_tx[mask].astype(np.float64)
        rows.append(
            "<tr>"
            f"<td>{t}</td>"
            f"<td>{n:,}</td>"
            f"<td>{float(np.median(v)):,.1f}</td>"
            f"<td>{float(np.mean(v)):,.1f}</td>"
            f"<td>{float(np.min(v)):,.1f}</td>"
            f"<td>{float(np.max(v)):,.1f}</td>"
            "</tr>"
        )
    return (
        "<p><strong>Summary per predicted type</strong> "
        "(raw total counts per cell — sum of the expression matrix)</p>"
        '<table class="genes">'
        "<tr><th>Predicted type</th><th>Cells</th><th>Median</th>"
        "<th>Mean</th><th>Min</th><th>Max</th></tr>" + "".join(rows) + "</table>"
    )


def _section_asset_links(sample_prefix: str) -> str:
    def name(base: str) -> str:
        return f"{sample_prefix}{base}" if sample_prefix else base

    items = [
        ("Cell type summary (PDF)", name("cell_type_summary.pdf")),
        ("Confidence (PDF)", name("confidence_analysis.pdf")),
        ("Spatial plots (PDF)", name("spatial_plots.pdf")),
        ("Transcripts vs classification (PDF)", name("transcript_vs_classification.pdf")),
        ("Xenium UMAP (PDF)", name("xenium_umap.pdf")),
        ("scRNA UMAP (PDF)", name("scrna_umap.pdf")),
        ("Interactive choropleth (HTML)", name("choropleth.html")),
    ]
    lis = "\n".join(f'<li><a href="{fn}">{label}</a></li>' for label, fn in items)
    return f'<div class="card"><ul class="links" style="list-style:none;padding-left:0">{lis}</ul></div>'
