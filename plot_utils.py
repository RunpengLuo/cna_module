import os
import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages

from utils import *


def build_ch_boundary(bb: pd.DataFrame):
    chrs = sort_chroms(bb["#CHR"].unique().tolist())
    chr_sizes = OrderedDict()
    for ch in chrs:
        chr_sizes[ch] = bb.loc[bb["#CHR"] == ch, "END"].max()

    # get 1d plot chromosome offsets
    chr_shift = int(5e6)  # slightly shift chr1 to right
    chr_offsets = OrderedDict()
    for i, ch in enumerate(chrs):
        if i == 0:
            chr_offsets[ch] = chr_shift
        else:
            prev_ch = chrs[i - 1]
            offset = chr_offsets[prev_ch] + chr_sizes[prev_ch]
            chr_offsets[ch] = offset
    chr_end = chr_offsets[chrs[-1]] + chr_sizes[chrs[-1]] + chr_shift
    chr_bounds = list(chr_offsets.values()) + [
        chr_offsets[chrs[-1]] + chr_sizes[chrs[-1]]
    ]

    # infer chromosome-gaps from SEG file
    chr_gaps = OrderedDict()
    for ch in chrs:
        # pre-sorted by start position before plot1d2d
        chr_regions = bb[bb["#CHR"] == ch][["START", "END"]].to_numpy()
        chr_regions_shift = chr_regions + chr_offsets[ch]
        chr_gaps[ch] = []
        if chr_regions[0, 0] > 0:
            chr_gaps[ch].append([chr_offsets[ch], chr_regions_shift[0, 0]])
        for i in range(len(chr_regions_shift) - 1):
            _, curr_t = chr_regions_shift[i,]
            next_s, _ = chr_regions_shift[i + 1,]
            if curr_t < next_s:
                chr_gaps[ch].append([curr_t, next_s])

    xlab_chrs = chrs  # ignore first dummy variable
    xtick_chrs = []
    for i in range(len(chrs)):
        l = chr_offsets[chrs[i]]
        if i < len(chrs) - 2:
            r = chr_offsets[chrs[i + 1]]
        else:
            r = chr_end
        xtick_chrs.append((l + r) / 2)

    # 1d position, global to all chromosome
    bb_positions = bb.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + (r.START + r.END) // 2, axis=1
    ).to_numpy()
    bb_starts = bb.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + r.START, axis=1
    ).to_numpy()
    bb_ends = bb.apply(func=lambda r: chr_offsets[r["#CHR"]] + r.END, axis=1).to_numpy()
    return (
        bb_positions,
        bb_starts,
        bb_ends,
        chr_bounds,
        chr_gaps,
        chr_end,
        xlab_chrs,
        xtick_chrs,
    )


def plot_1d2d(
    bb_dir: str,
    out_dir: str,
    out_prefix="",
    plot_normal=False,
    clusters=None,
    expected_rdrs=None,
    expected_bafs=None,
):
    os.makedirs(out_dir, exist_ok=True)
    bin_file = os.path.join(bb_dir, "bin_position.tsv.gz")
    cov_matrix = os.path.join(bb_dir, "bin_matrix.cov.npz")
    baf_matrix = os.path.join(bb_dir, "bin_matrix.baf.npz")
    rdr_matrix = os.path.join(bb_dir, "bin_matrix.rdr_corr.npz")

    bb = pd.read_table(bin_file, sep="\t")
    cov_mat = np.load(cov_matrix)["mat"].astype(np.float64)
    baf_mat = np.load(baf_matrix)["mat"].astype(np.float64)
    rdr_mat = np.load(rdr_matrix)["mat"].astype(np.float64)

    nsamples = baf_mat.shape[1]
    samples = ["normal"] + [f"tumor{i}" for i in range(1, nsamples)]

    ########################################
    ret = build_ch_boundary(bb)
    (
        bb_positions,
        bb_starts,
        bb_ends,
        chr_bounds,
        chr_gaps,
        chr_end,
        xlab_chrs,
        xtick_chrs,
    ) = ret

    sns.set_style("whitegrid")
    if clusters is None:
        clusters = np.ones(len(bb))
    clusters_hue = clusters
    cluster_ids = sorted(list(np.unique(clusters)))
    num_cluster = len(cluster_ids)
    if num_cluster > 8:
        palette = sns.color_palette("husl", n_colors=num_cluster)
    else:
        palette = sns.color_palette("Set2", n_colors=num_cluster)
    sns.set_palette(palette)

    markersize = float(max(2, 4 - np.floor(len(bb) / 500)))
    markersize_centroid = 10
    marker_bd_width = 0.8
    rdr_linewidth = 1.5
    baf_linewidth = 1.5
    minRDR = np.floor(np.min(rdr_mat))
    maxRDR = np.ceil(np.max(rdr_mat))

    assert samples[0] == "normal"
    for si, sample in enumerate(samples):
        if si == 0 and not plot_normal:
            print(f"skip {sample}")
            continue
        print(f"plot {sample}")
        out1d = os.path.join(out_dir, f"{out_prefix}{sample}.1D.png")
        out2d = os.path.join(out_dir, f"{out_prefix}{sample}.2D.png")
        covs = cov_mat[:, si]
        bafs = baf_mat[:, si]
        if si == 0:
            rdrs = np.ones(len(bb_positions))
        else:
            rdrs = rdr_mat[:, si - 1]

        ########################################
        if not clusters is None:
            g0 = sns.JointGrid(
                x=bafs,
                y=rdrs,
                hue=clusters_hue,
                palette=palette,
            )
        else:
            g0 = sns.JointGrid(
                x=bafs,
                y=rdrs,
            )

        g0.refline(x=0.50)
        g0.plot_joint(sns.scatterplot, s=markersize, legend=False, edgecolors="none")
        if si > 0:
            g0.plot_marginals(
                sns.histplot,
                kde=True,
                common_norm=False,
                stat="density",
            )
        scatter = g0.ax_joint.collections[0]
        colors_ = scatter.get_facecolors()

        if not expected_bafs is None and not expected_rdrs is None:
            exp_bafs = expected_bafs[:, si]
            exp_rdrs = expected_rdrs[:, si - 1] if si > 0 else np.ones(len(exp_bafs))
            for ci, cluster_id in enumerate(cluster_ids):
                center_text = str(cluster_id)
                fontdict = {"fontsize": 10}
                g0.ax_joint.text(exp_bafs[ci], exp_rdrs[ci], center_text, fontdict)
            g0.ax_joint.scatter(
                x=exp_bafs,
                y=exp_rdrs,
                facecolors="none",
                edgecolors="black",
                s=markersize_centroid,
                linewidth=marker_bd_width,
            )
        g0.set_axis_labels(xlabel="mhBAF", ylabel="RDR")
        g0.figure.suptitle(sample)
        plt.tight_layout()
        g0.savefig(out2d, dpi=300, format="png")
        plt.close(g0.figure)

        ########################################
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 4))
        gs = []
        for ai, y in enumerate([rdrs, bafs, covs]):
            if ai == 0 and not clusters is None:
                g = sns.scatterplot(
                    x=bb_positions,
                    y=y,
                    ax=axes[ai],
                    s=markersize,
                    color=colors_,
                    hue=clusters_hue,
                    palette=palette,
                )
                axes[0].legend(markerscale=6)
                sns.move_legend(
                    axes[0], "upper left", bbox_to_anchor=(1, 1), title=None
                )
            else:
                g = sns.scatterplot(
                    x=bb_positions, y=y, ax=axes[ai], s=markersize, color=colors_
                )
            gs.append(g)
        gs[0].collections[0].set_facecolors(colors_)
        minBAF_, maxBAF_ = axes[1].get_ylim()
        for ai in range(3):
            # add chromosome boundary
            axes[ai].vlines(
                chr_bounds,
                ymin=0,
                ymax=1,
                transform=axes[ai].get_xaxis_transform(),
                linewidth=0.5,
                colors="k",
            )

            # add centromere boxes
            for ch, gaps in chr_gaps.items():
                for gap in gaps:
                    axes[ai].add_patch(
                        Rectangle(
                            (gap[0], 0),
                            gap[1] - gap[0],
                            1,
                            linewidth=0,
                            color=(0, 0, 0, 0.2),
                            transform=axes[ai].get_xaxis_transform(),
                        )
                    )
        if not expected_bafs is None and not expected_rdrs is None:
            rdr_lines = []
            baf_lines = []
            bl_colors = [(0, 0, 0, 1)] * len(clusters)
            for ci, cluster_id in enumerate(cluster_ids):
                exp_baf = expected_bafs[int(ci), si]
                exp_rdr = expected_rdrs[int(ci), si - 1] if si > 0 else 1.0
                my_starts = bb_starts[clusters == cluster_id]
                my_ends = bb_ends[clusters == cluster_id]
                rdr_lines.extend(
                    [
                        [(my_starts[bi], exp_rdr), (my_ends[bi], exp_rdr)]
                        for bi in range(len(my_starts))
                    ]
                )
                baf_lines.extend(
                    [
                        [(my_starts[bi], exp_baf), (my_ends[bi], exp_baf)]
                        for bi in range(len(my_starts))
                    ]
                )
            axes[0].add_collection(
                LineCollection(rdr_lines, linewidth=rdr_linewidth, colors=bl_colors)
            )
            axes[1].add_collection(
                LineCollection(baf_lines, linewidth=baf_linewidth, colors=bl_colors)
            )

        # add BAF 0.5 line
        axes[1].hlines(
            y=0.5,
            xmin=0,
            xmax=chr_end,
            colors="grey",
            linestyle=":",
            linewidth=1,
        )

        axes[0].grid(False)
        axes[0].set_ylim(minRDR, maxRDR)
        axes[0].set_ylabel("RDR")
        axes[0].title.set_text(sample)

        axes[1].grid(False)
        axes[1].set_ylim(minBAF_, maxBAF_)
        axes[1].set_ylabel("mhBAF")
        axes[1].title.set_text("")

        axes[2].grid(False)
        axes[2].set_ylabel("COV")
        axes[2].title.set_text("")

        plt.setp(axes, xlim=(0, chr_end), xticks=xtick_chrs, xlabel="")
        for ai in range(3):
            axes[ai].set_xticklabels(xlab_chrs, rotation=60, fontsize=8)
        # sns.move_legend(axes[0], "upper left", bbox_to_anchor=(1, 1), title=None)
        plt.tight_layout()
        fig.savefig(out1d, dpi=300, format="png")
        plt.close(fig)
    return
