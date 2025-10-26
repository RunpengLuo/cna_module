import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pybedtools import BedTool

import matplotlib.pyplot as plt
import seaborn as sns

def compute_gc_content(bin_info: pd.DataFrame, ref_file: str, map_file=None):
    print("compute GC content")
    gc_df = bin_info.merge(
        BedTool.from_dataframe(bin_info[["#CHR", "START", "END"]])
        .nucleotide_content(fi=ref_file)
        .to_dataframe(disable_auto_names=True)
        .rename(
            columns={
                "#1_usercol": "#CHR",
                "2_usercol": "START",
                "3_usercol": "END",
                "5_pct_gc": "GC",
            }
        )[["#CHR", "START", "END", "GC"]]
    )
    gc_df["GC"] /= 100  # convert to %GC

    if map_file:
        print("compute mappability")
        # Intersect with mappability bedGraph to get per-bin mean score
        map_bt = BedTool(map_file)
        map_cov = (
            BedTool.from_dataframe(bin_info[["#CHR", "START", "END"]])
            .map(b=map_bt, c=4, o="mean")
            .to_dataframe(disable_auto_names=True)
            .rename(
                columns={
                    "#1_usercol": "#CHR",
                    "2_usercol": "START",
                    "3_usercol": "END",
                    "4": "MAP"
                }
            )
        )
        gc_df = gc_df.merge(map_cov, on=["#CHR", "START", "END"], how="left")
        gc_df["MAP"] = gc_df["MAP"].fillna(1.0)
    else:
        gc_df["MAP"] = 1.0

    return gc_df

def compute_RDR(
    bin_ids: np.ndarray,
    bin_info: pd.DataFrame,
    snp_info: pd.DataFrame,
    dp_mat: np.ndarray,
    nbins: int,
    ntumor_samples: int,
    read_type: str,
    correct_gc=True,
    ref_file=None,
    map_file=None,
    out_dir=None,
    grp_id="bin_id",
    min_map=0.9
):
    print("compute RDR")

    # depth is fractional, #bases should be integer.
    snp_bss = snp_info["BLOCKSIZE"].to_numpy()
    bases_mat = np.ceil(dp_mat * snp_bss[:, None]).astype(np.int64)

    total_bases = np.sum(bases_mat, axis=0)
    total_bases_normal = total_bases[0]
    total_bases_tumors = total_bases[1:]
    library_correction = total_bases_normal / total_bases_tumors
    print(f"RDR library normalization factor: {library_correction}")

    snp_grp_bins = snp_info.groupby(by=grp_id, sort=False)
    bin_bases_mat = np.zeros((nbins, 1 + ntumor_samples), dtype=np.int64)
    for bi, bin_id in enumerate(bin_ids):
        snp_bin = snp_grp_bins.get_group(bin_id)
        snp_bin_idx = snp_bin.index.to_numpy()
        bin_bases_mat[bi, :] = np.sum(bases_mat[snp_bin_idx], axis=0)

    # read-depth normalized by bin length
    bin_bss = bin_info["BLOCKSIZE"].to_numpy()
    bin_depth_mat = bin_bases_mat / bin_bss[:, None]

    raw_rdr_mat = bin_depth_mat[:, 1:] / bin_depth_mat[:, 0][:, None]
    raw_rdr_mat *= library_correction[None, :]
    if correct_gc:
        print("correct for GC biases")
        gc_df = compute_gc_content(bin_info, ref_file, map_file)
        gc = gc_df["GC"].to_numpy()
        gccorr_rdr_mat = np.zeros_like(raw_rdr_mat, dtype=np.float32)
        for si in range(ntumor_samples):
            # gc-correction via median quantile regression
            gc_df = pd.DataFrame({"RD": raw_rdr_mat[:, si], "GC": gc})
            mod = smf.quantreg("RD ~ GC + I(GC**2)", data=gc_df).fit(q=0.5)
            gc_df["GCCORR"] = mod.predict(gc_df)
            corr_rdrs = gc_df["RD"] / gc_df["GCCORR"].where(
                (gc_df["GCCORR"] > 0) & ~pd.isnull(gc_df["GCCORR"]), 1
            )
            gccorr_rdr_mat[:, si] = corr_rdrs / np.mean(corr_rdrs)
            plot_gc_bias(gc, raw_rdr_mat[:, si], gccorr_rdr_mat[:, si], sample_id=f"tumor{si}", out_dir=out_dir)
        raw_rdr_mat = gccorr_rdr_mat
    rdr_mat = raw_rdr_mat
    return rdr_mat


def plot_gc_bias(gc, raw_rdr, corr_rdr, sample_id=None, out_dir=None):
    def mad(x):
        return np.median(np.abs(x - np.median(x)))

    mad_raw = mad(raw_rdr)
    mad_corr = mad(corr_rdr)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

    # --- Uncorrected ---
    hb1 = axes[0].hexbin(
        gc, raw_rdr, gridsize=100, cmap="Blues",
        mincnt=1, linewidths=0, reduce_C_function=np.median
    )
    axes[0].set_xlabel("GC content")
    axes[0].set_ylabel("RDR")
    axes[0].set_title(f"Uncorrected RDR\nMAD={mad_raw:.3f}")

    # --- Corrected ---
    hb2 = axes[1].hexbin(
        gc, corr_rdr, gridsize=100, cmap="Blues",
        mincnt=1, linewidths=0, reduce_C_function=np.median
    )
    axes[1].set_xlabel("GC content")
    axes[1].set_ylabel("RDR")
    axes[1].set_title(f"GC-corrected RDR\nMAD={mad_corr:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{sample_id}.gc_corr.png"), dpi=300)
    plt.close(fig)
