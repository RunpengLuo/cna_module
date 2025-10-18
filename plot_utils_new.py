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
from matplotlib.patches import Ellipse


from utils import *

##################################################
def build_ch_boundary(
    region_df: pd.DataFrame, chrs: list, chr_sizes: dict, chr_shift=int(10e6)
):
    # get 1d plot chromosome offsets, global information
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
    xlab_chrs = chrs  # ignore first dummy variable
    xtick_chrs = []
    for i in range(len(chrs)):
        l = chr_offsets[chrs[i]]
        if i < len(chrs) - 2:
            r = chr_offsets[chrs[i + 1]]
        else:
            r = chr_end
        xtick_chrs.append((l + r) / 2)

    # infer chromosome-gaps from SEG file
    # all samples should share same gaps
    dummy_sample = region_df["SAMPLE"].unique()[0]
    chr_gaps = OrderedDict()
    for ch in chrs:
        chr_regions = region_df[(region_df["#CHR"] == ch) & (region_df["SAMPLE"] == dummy_sample)][["START", "END"]].to_numpy()
        chr_regions_shift = chr_regions + chr_offsets[ch]
        chr_gaps[ch] = []
        if chr_regions[0, 0] > 0:
            chr_gaps[ch].append([chr_offsets[ch], chr_regions_shift[0, 0]])
        for i in range(len(chr_regions_shift) - 1):
            _, curr_t = chr_regions_shift[i,]
            next_s, next_t = chr_regions_shift[i + 1,]
            if curr_t < next_s:
                chr_gaps[ch].append([curr_t, next_s])
        if next_t - chr_offsets[ch] < chr_sizes[ch]:
            chr_gaps[ch].append([next_t, chr_offsets[ch] + chr_sizes[ch]])
    return (
        chr_offsets,
        chr_bounds,
        chr_gaps,
        chr_end,
        xlab_chrs,
        xtick_chrs,
    )


def plot_1d2d(
    block_info: pd.DataFrame,
    baf_mat: np.ndarray,
    rdr_mat: np.ndarray,
    cluster_labels: np.ndarray,
    expected_rdrs: np.ndarray,
    expected_bafs: np.ndarray,
    genome_file: str,
    out_dir: str,
    out_prefix="",
    plot_mirror_baf=False
):
    chrom_sizes = get_chr_sizes(genome_file)
    chrs = block_info["#CHR"].unique().tolist()
    block_info["SAMPLE"] = "tumor"
    ret = build_ch_boundary(block_info, chrs, chrom_sizes, chr_shift=int(10e6))
    (
        chr_offsets,
        chr_bounds,
        chr_gaps,
        chr_end,
        xlab_chrs,
        xtick_chrs,
    ) = ret

    positions = block_info.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + (r.START + r.END) // 2, axis=1
    ).to_numpy()
    abs_starts = block_info.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + r.START, axis=1
    ).to_numpy()
    abs_ends = block_info.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + r.END, axis=1
    ).to_numpy()

    sns.set_style("whitegrid")
    uniq_cluster_labels = np.sort(np.unique(cluster_labels))
    num_colors = len(np.unique(cluster_labels))
    if num_colors > 8:
        palette = sns.color_palette("husl", n_colors=num_colors)
    else:
        palette = sns.color_palette("Set2", n_colors=num_colors)
    sns.set_palette(palette)

    markersize = float(max(2, 4 - np.floor(len(block_info) / 500)))
    markersize_centroid = 10
    marker_bd_width = 0.8
    rdr_linewidth = 1.5
    baf_linewidth = 1.5
    minRDR = np.floor(np.min(rdr_mat))
    maxRDR = np.ceil(np.max(rdr_mat))

    samples = ["tumor"]
    for si, sample in enumerate(samples):
        out1d = os.path.join(out_dir, f"{out_prefix}{sample}.1D.png")
        out2d = os.path.join(out_dir, f"{out_prefix}{sample}.2D.png")

        bafs = baf_mat[:, si]
        rdrs = rdr_mat[:, si]
        
        g0 = sns.JointGrid(
            x=bafs,
            y=rdrs,
            hue=cluster_labels,
            palette=palette,
        )
        g0.refline(x=0.50)
        g0.plot_joint(sns.scatterplot, s=markersize, legend=False, edgecolors="none")
        g0.plot_marginals(
            sns.histplot,
            kde=True,
            common_norm=False,
            bins=50
            # stat="density",
            # element="step"
        )
        scatter = g0.ax_joint.collections[0]
        colors_ = scatter.get_facecolors()

        exp_bafs = expected_bafs[:, si]
        exp_rdrs = expected_rdrs[:, si]
        for ci, cluster_id in enumerate(uniq_cluster_labels):
            center_text = str(cluster_id)
            fontdict = {"fontsize": 10}
            g0.ax_joint.text(exp_bafs[ci], exp_rdrs[ci], center_text, fontdict)
            if plot_mirror_baf:
                g0.ax_joint.text(1 - exp_bafs[ci], exp_rdrs[ci], center_text, fontdict)
        g0.ax_joint.scatter(
            x=exp_bafs,
            y=exp_rdrs,
            facecolors="none",
            edgecolors="black",
            s=markersize_centroid,
            linewidth=marker_bd_width,
        )
        if plot_mirror_baf:
            g0.ax_joint.scatter(
                x=1 - exp_bafs,
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
        

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 8))
        gs = []
        plot_mats = [rdrs, bafs]
        for ai, y in enumerate(plot_mats):
            if ai == 0:
                g = sns.scatterplot(
                    x=positions,
                    y=y,
                    ax=axes[ai],
                    s=markersize,
                    color=colors_,
                    hue=cluster_labels,
                    palette=palette,
                )
                axes[0].legend(markerscale=6)
                sns.move_legend(
                    axes[0], "upper left", bbox_to_anchor=(1, 1), title=None
                )
            else:
                g = sns.scatterplot(
                    x=positions, y=y, ax=axes[ai], s=markersize, color=colors_
                )
            gs.append(g)
        
        gs[0].collections[0].set_facecolors(colors_)
        minBAF_, maxBAF_ = axes[1].get_ylim()
        axes[0].set_ylim(minRDR, maxRDR)
        axes[0].set_ylabel("RDR")
        axes[0].title.set_text(sample)
        for yv in np.arange(minRDR, maxRDR, 1):
            axes[0].hlines(
                y=yv,
                xmin=0,
                xmax=chr_end,
                colors="grey",
                linewidth=0.5,
                alpha=0.5
            )
        
        axes[1].set_ylabel("mhBAF")
        axes[1].title.set_text("")
        if not plot_mirror_baf:
            # add BAF 0.5 line
            axes[1].hlines(
                y=0.5,
                xmin=0,
                xmax=chr_end,
                colors="grey",
                linestyle=":",
                linewidth=1,
            )
            axes[1].set_ylim(0, 0.51)
            yticks = np.arange(0.0, 0.51, 0.1)
            axes[1].set_yticks(yticks)
            axes[1].set_yticklabels([f"{y:.1f}" for y in yticks])
            for yv in np.arange(0.0, 0.41, 0.1):
                axes[1].hlines(
                    y=yv,
                    xmin=0,
                    xmax=chr_end,
                    colors="grey",
                    linewidth=0.5,
                    alpha=0.5
                )
        else:
            axes[1].set_ylim(0, 1.01)
            yticks = [0.0, 0.25, 0.50, 0.75, 1.0]
            axes[1].set_yticks(yticks)
            axes[1].set_yticklabels([f"{y:.2f}" for y in yticks])
            for yv in np.arange(0.0, 1.01, 0.1):
                axes[1].hlines(
                    y=yv,
                    xmin=0,
                    xmax=chr_end,
                    colors="grey",
                    linewidth=0.5,
                    alpha=0.5
                )
        
        for ai in range(len(gs)):
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
                    if gap[1] - gap[0] < 10e6:
                        continue
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
        
        rdr_lines = []
        baf_lines = []
        bl_colors = [(0, 0, 0, 1)] * len(cluster_labels)
        for ci, cluster_id in enumerate(uniq_cluster_labels):
            exp_baf = expected_bafs[int(ci), si]
            exp_rdr = expected_rdrs[int(ci), si]
            my_starts = abs_starts[cluster_labels == cluster_id]
            my_ends = abs_ends[cluster_labels == cluster_id]
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
            if not plot_mirror_baf: # also plot mirrored BAF lines
                exp_baf2 = 1 - exp_baf
                baf_lines.extend(
                    [
                        [(my_starts[bi], exp_baf2), (my_ends[bi], exp_baf2)]
                        for bi in range(len(my_starts))
                    ]
                )
        axes[0].add_collection(
            LineCollection(rdr_lines, linewidth=rdr_linewidth, colors=bl_colors)
        )
        axes[1].add_collection(
            LineCollection(baf_lines, linewidth=baf_linewidth, colors=bl_colors)
        )
        plt.setp(axes, xlim=(0, chr_end), xticks=xtick_chrs, xlabel="")
        for ai in range(len(gs)):
            axes[ai].set_xticklabels(xlab_chrs, rotation=60, fontsize=8)
            if ai < 2:
                axes[ai].grid(False)
        # sns.move_legend(axes[0], "upper left", bbox_to_anchor=(1, 1), title=None)
        plt.tight_layout()
        fig.savefig(out1d, dpi=300, format="png")
        plt.close(fig)
    return


