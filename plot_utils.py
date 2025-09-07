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

def add_gaussian_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    # Eigen-decompose covariance to get principal axes
    vals, vecs = np.linalg.eigh(cov)
    # sort large -> small
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # width/height: 2*n_std*sqrt(eigenvals)
    width, height = 2 * n_std * np.sqrt(vals)
    # angle in degrees from x-axis of the largest eigenvector
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ell = Ellipse(
        xy=mean, width=width, height=height, angle=angle,
        fill=True, lw=2, color="black", **kwargs
    )
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.1)
    ax.add_patch(ell)


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
    fitted_means=None,
    fitted_covs=None,
    mirrored_baf=False,
    plot_potts=False,
    plot_alpha_beta=False,
    plot_cov=False,
    baf_file="bin_matrix.baf.npz",
    rdr_file="bin_matrix.rdr.npz",
    potts_states=None
):
    print("plot 1d2d BAF and RDRs")
    os.makedirs(out_dir, exist_ok=True)
    bb = pd.read_table(os.path.join(bb_dir, "bin_info.tsv.gz"), sep="\t")
    baf_mat = np.load(os.path.join(bb_dir, baf_file))["mat"].astype(np.float64)
    rdr_mat = np.load(os.path.join(bb_dir, rdr_file))["mat"].astype(np.float64)
    # if plot_potts:
    #     potts_states = np.load(os.path.join(bb_dir, "bin_matrix.potts.npz"))["mat"].astype(np.int8)
    
    nrow_1d = 2
    if plot_cov:
        nrow_1d += 1
        cov_mat = np.load(os.path.join(bb_dir, "bin_matrix.cov.npz"))["mat"].astype(np.float64)
    
    if plot_alpha_beta:
        assert plot_cov
        nrow_1d += 2
        alpha_mat = np.load(os.path.join(bb_dir, "bin_matrix.alpha.npz"))["mat"].astype(np.int32)
        beta_mat = np.load(os.path.join(bb_dir, "bin_matrix.beta.npz"))["mat"].astype(np.int32)

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

    si2cluster_map = {i: 0 for i in range(nsamples)} # all samples use first set of cluster id, by default.
    if plot_potts and not clusters is None:
        raise ValueError("cannot plot both potts and clusters")
    else:
        if not clusters is None:
            assert len(clusters) == len(bb)
            print(f"plot given clusters")
            clusters = clusters.reshape(1, len(clusters))
        if plot_potts:
            print(f"plot potts labels")
            clusters = potts_states.T # (2, nbins)
            for i in range(1, nsamples):
                si2cluster_map[i] = 1 # all tumor samples use tumor-specific potts labels.
    if clusters is None:
        clusters = np.ones((1, len(bb)))
    num_colors = max([len(np.unique(clusters[i])) for i in range(clusters.shape[0])])
    if num_colors > 8:
        palette = sns.color_palette("husl", n_colors=num_colors)
    else:
        palette = sns.color_palette("Set2", n_colors=num_colors)
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

        if plot_cov:
            covs = cov_mat[:, si]
        if plot_alpha_beta:
            alphas = alpha_mat[:, si]
            betas = beta_mat[:, si]

        bafs = baf_mat[:, si]
        if si == 0:
            rdrs = np.ones(len(bb_positions))
        else:
            rdrs = rdr_mat[:, si - 1]
        
        si_clusters = clusters[si2cluster_map[si]]
        si_cluster_ids = np.unique(si_clusters)

        ########################################
        if si == 0: # normal sample
            fig, ax = plt.subplots(1, 1)
            g0 = sns.histplot(x=bafs, ax=ax, bins=50, binrange=[0, 1])
            ax.vlines(
                0.5,
                ymin=0,
                ymax=1,
                transform=ax.get_xaxis_transform(),
                linewidth=0.5,
                colors="k",
            )
            mu_baf = np.mean(bafs)
            std_baf = np.std(bafs)
            med_baf = np.median(bafs)
            ax.set_title(f"{sample} mu={mu_baf:.3f} std={std_baf:.3f} med={med_baf:.3f}")
            ax.set_xlabel(xlabel="mhBAF")
            ax.grid(False)
            colors_ = None
            plt.tight_layout()
            plt.savefig(out2d, dpi=300, format="png")
            plt.close(fig)
        else:
            g0 = sns.JointGrid(
                x=bafs,
                y=rdrs,
                hue=si_clusters,
                palette=palette,
            )
            g0.refline(x=0.50)
            g0.plot_joint(sns.scatterplot, s=markersize, legend=False, edgecolors="none")
            g0.plot_marginals(
                sns.histplot,
                kde=True,
                common_norm=False,
                # stat="density",
                # element="step"
            )
            scatter = g0.ax_joint.collections[0]
            colors_ = scatter.get_facecolors()
            
            # if not fitted_means is None:
            #     for k in range(fitted_means.shape[0]):
            #         print(fitted_means)
            #         print(fitted_covs)
            #         add_gaussian_ellipse(g0.ax_joint, fitted_means[k], fitted_covs[k])

            if not expected_bafs is None and not expected_rdrs is None:
                exp_bafs = expected_bafs[:, si]
                exp_rdrs = expected_rdrs[:, si - 1] if si > 0 else np.ones(len(exp_bafs))
                for ci, cluster_id in enumerate(si_cluster_ids):
                    center_text = str(cluster_id)
                    fontdict = {"fontsize": 10}
                    g0.ax_joint.text(exp_bafs[ci], exp_rdrs[ci], center_text, fontdict)
                    if not mirrored_baf:
                        g0.ax_joint.text(1 - exp_bafs[ci], exp_rdrs[ci], center_text, fontdict)
                g0.ax_joint.scatter(
                    x=exp_bafs,
                    y=exp_rdrs,
                    facecolors="none",
                    edgecolors="black",
                    s=markersize_centroid,
                    linewidth=marker_bd_width,
                )
                if not mirrored_baf:
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

        ########################################
        if nrow_1d < 3:
            figsize=(20, 8)
        else:
            figsize=(20, 1.5 * nrow_1d)
        fig, axes = plt.subplots(nrows=nrow_1d, ncols=1, figsize=figsize)
        gs = []
        plot_mats = [rdrs, bafs]
        if plot_cov:
            plot_mats.append(covs)
        if plot_alpha_beta:
            plot_mats.extend([alphas, betas])
        for ai, y in enumerate(plot_mats):
            if ai == 0:
                g = sns.scatterplot(
                    x=bb_positions,
                    y=y,
                    ax=axes[ai],
                    s=markersize,
                    color=colors_,
                    hue=si_clusters,
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
        if mirrored_baf:
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

        if plot_cov:
            axes[2].set_ylabel("COV")
            axes[2].title.set_text("")
        if plot_alpha_beta:
            axes[3].set_ylabel("ALPHA")
            axes[3].title.set_text("")
            axes[4].set_ylabel("BETA")
            axes[4].title.set_text("")

        for ai in range(nrow_1d):
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
        if not expected_bafs is None and not expected_rdrs is None:
            rdr_lines = []
            baf_lines = []
            bl_colors = [(0, 0, 0, 1)] * len(si_clusters)
            for ci, cluster_id in enumerate(si_cluster_ids):
                exp_baf = expected_bafs[int(ci), si]
                exp_rdr = expected_rdrs[int(ci), si - 1] if si > 0 else 1.0
                my_starts = bb_starts[si_clusters == cluster_id]
                my_ends = bb_ends[si_clusters == cluster_id]
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
                if not mirrored_baf: # also plot mirrored BAF lines
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
        for ai in range(nrow_1d):
            axes[ai].set_xticklabels(xlab_chrs, rotation=60, fontsize=8)
            if ai < 2:
                axes[ai].grid(False)
        # sns.move_legend(axes[0], "upper left", bbox_to_anchor=(1, 1), title=None)
        plt.tight_layout()
        fig.savefig(out1d, dpi=300, format="png")
        plt.close(fig)
    return

def plot_2d(X: np.ndarray, labels=None, means=None, title="labels", height=3, s=3, out_file=None):
    g0 = sns.JointGrid(
        x=X[:, 0],
        y=X[:, 1],
        hue=labels,
        palette=sns.color_palette("Set2", n_colors=len(np.unique(labels))),
        height=height
    )
    g0.refline(x=0.50)
    g0.plot_joint(sns.scatterplot, s=s, legend=False, edgecolors="none")
    g0.plot_marginals(
        sns.histplot,
        kde=False,
        common_norm=False,
        # stat="density",
        # element="step"
    )
    scatter = g0.ax_joint.collections[0]
    colors_ = scatter.get_facecolors()  # reused in 1D plot
    scatter.set_facecolors(colors_)
    if not means is None:
        g0.ax_joint.scatter(x=means[:, 0], y=means[:, 1], marker="x", c="black")
    g0.set_axis_labels(xlabel="mhBAF", ylabel="RDR")
    g0.figure.suptitle(title)
    plt.tight_layout()
    if not out_file is None:
        g0.savefig(out_file, dpi=300, format="png")
        plt.close(g0.figure)

if __name__ == "__main__":
    _, bb_dir = sys.argv
    plot_1d2d(
        bb_dir,
        bb_dir,
        plot_normal=True,
        clusters=None,
        expected_rdrs=None,
        expected_bafs=None,
        plot_cov=True,
        plot_alpha_beta=True
    )
