import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d

from pybedtools import BedTool

import matplotlib.pyplot as plt
import seaborn as sns

def compute_gc_content(bin_info: pd.DataFrame, ref_file: str, mapp_file=None, genome_file=None):
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
    if mapp_file:
        print("compute mappability")
        map_bt = BedTool(mapp_file)
        map_cov = (
            BedTool.from_dataframe(bin_info[["#CHR", "START", "END"]])
            .map(b=map_bt, c=4, o="mean", g=genome_file)
            .to_dataframe(disable_auto_names=True)
        )
        map_cov.columns = ["#CHR", "START", "END", "MAP"]
        map_cov["MAP"] = pd.to_numeric(map_cov["MAP"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
        gc_df = gc_df.merge(map_cov, on=["#CHR", "START", "END"], how="left")
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
    mapp_file=None,
    genome_file=None,
    out_dir=None,
    grp_id="bin_id",
    map_cutoff=0.9
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
        gc_df = compute_gc_content(bin_info, ref_file, mapp_file, genome_file)
        raw_rdr_mat = bias_correction_rdr(raw_rdr_mat, gc_df, read_type, out_dir)
        # raw_rdr_mat = bias_correction_hmmcopy(bin_depth_mat, gc_df, read_type, map_cutoff)
    rdr_mat = raw_rdr_mat
    return rdr_mat

def bias_correction_rdr(raw_rdr_mat: np.ndarray, gc_df: pd.DataFrame, read_type: str, out_dir=None):
    print("correct for GC biases on RDR")
    gc = gc_df["GC"].to_numpy()
    mapv = gc_df["MAP"].to_numpy()
    gccorr_rdr_mat = np.zeros_like(raw_rdr_mat, dtype=np.float32)
    for si in range(raw_rdr_mat.shape[1]):
        gc_df = pd.DataFrame({
            "RD": raw_rdr_mat[:, si],
            "GC": gc,
            "MAP": mapv
        })
        if read_type != "TGS" and np.any(gc_df["MAP"] != 1):
            mod = smf.quantreg("RD ~ GC + I(GC**2) + MAP + I(MAP**2)", data=gc_df).fit(q=0.5)
            gc_df["CORR_FIT"] = mod.predict(gc_df[["GC", "MAP"]])
            corr_rdrs = gc_df["RD"] / gc_df["CORR_FIT"].where(
                (gc_df["CORR_FIT"] > 0) & ~pd.isnull(gc_df["CORR_FIT"]), 1
            )
        else:
            mod = smf.quantreg("RD ~ GC + I(GC**2)", data=gc_df).fit(q=0.5)
            gc_df["CORR_FIT"] = mod.predict(gc_df[["GC"]])
            corr_rdrs = gc_df["RD"] / gc_df["CORR_FIT"].where(
                (gc_df["CORR_FIT"] > 0) & ~pd.isnull(gc_df["CORR_FIT"]), 1
            )
        corr_factor = np.mean(corr_rdrs)
        print(f"correction factor={corr_factor}")
        gccorr_rdr_mat[:, si] = corr_rdrs / corr_factor
        plot_gc_bias(gc, raw_rdr_mat[:, si], gccorr_rdr_mat[:, si], sample_id=f"tumor{si}", out_dir=out_dir)
    return gccorr_rdr_mat


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


# def bias_correction_hmmcopy(
#     bin_depth_mat: np.ndarray, 
#     gc_df: pd.DataFrame, 
#     read_type: str,
#     map_cutoff=0.9,
#     sample_size=50000
# ):
#     print("correct for GC biases using HMMCopy method")
#     nbins, nsamples = bin_depth_mat.shape
#     gc = gc_df["GC"].to_numpy()
#     mapv = gc_df["MAP"].to_numpy()
#     corrected_mat = np.zeros_like(bin_depth_mat, dtype=np.float32)
#     for si in range(nsamples):
#         sgc_df = pd.DataFrame({
#             "reads": bin_depth_mat[:, si], "GC": gc, "MAP": mapv})
#         print(f"sample {si}, average depth={sgc_df.reads.mean():.3f} median depth={sgc_df.reads.median():.3f}")
#         corrected = correct_readcount(sgc_df, map_cutoff=map_cutoff)
#         corrected_mat[:, si] = corrected["cor_map"]
#     raw_rdr_mat = corrected_mat[:, 1:] / corrected_mat[:, 0][:, None]
#     return raw_rdr_mat

# def correct_readcount(gc_df: pd.DataFrame, map_cutoff=0.90, sample_size=50000,
#                       routlier = 0.01, doutlier = 0.001, coutlier = 0.01):
#     """
#     reads, GC, MAP
#     """
#     nbins = len(gc_df)
#     sample_size = min(sample_size, nbins)
#     gc_df["valid"] = True
#     gc_df["ideal"] = True

#     gc_df.loc[(gc_df["GC"] < 0.0) | (gc_df["reads"] <= 0), "valid"] = False
#     num_valids = gc_df["valid"].sum()
#     print(f"#valid bins={num_valids}/{nbins}")
#     if num_valids == 0:
#         raise ValueError("failed to perform GC correction")

#     reads_valid = gc_df.loc[gc_df["valid"], "reads"]
#     read_range = np.nanquantile(reads_valid, [0, 1 - routlier])

#     gc_valid = gc_df.loc[gc_df["valid"], "GC"]
#     gc_domain = np.nanquantile(gc_valid, [doutlier, 1 - doutlier])

#     gc_df.loc[
#         (~gc_df["valid"]) |
#         (gc_df["MAP"] < map_cutoff) |
#         (gc_df["reads"] <= read_range[0]) |
#         (gc_df["reads"] > read_range[1]) |
#         (gc_df["GC"] < gc_domain[0]) |
#         (gc_df["GC"] > gc_domain[1]),
#         "ideal"
#     ] = False
#     num_ideals = gc_df["ideal"].sum()
#     print(f"#ideal bins={num_ideals}/{nbins}")

#     ideal_idx = np.where(gc_df["ideal"])[0]
#     if len(ideal_idx) == 0:
#         raise ValueError("No ideal bins found for GC correction.")
#     select_idx = np.random.choice(ideal_idx, min(len(ideal_idx), sample_size), replace=False)

#     reads_sel = gc_df.loc[select_idx, "reads"].to_numpy()
#     gc_sel = gc_df.loc[select_idx, "GC"].to_numpy()
#     rough_fit = lowess(reads_sel, gc_sel, frac=0.03, return_sorted=True)

#     i = np.linspace(0, 1, 1001)
#     rough_interp = interp1d(rough_fit[:, 0], rough_fit[:, 1],
#                         bounds_error=False, fill_value="extrapolate")
#     rough_pred = rough_interp(i)
#     final_fit = lowess(rough_pred, i, frac=0.3, return_sorted=True)
#     final_interp = interp1d(final_fit[:, 0], final_fit[:, 1],
#                             bounds_error=False, fill_value="extrapolate")
#     gc_df["cor_gc"] = gc_df["reads"] / np.clip(final_interp(gc_df["GC"]), 1e-8, None)

#     cor_gc_valid = gc_df.loc[gc_df["valid"], "cor_gc"]
#     cor_range = np.nanquantile(cor_gc_valid, [0, 1 - coutlier])

#     map_set = gc_df.index[gc_df["cor_gc"] < cor_range[1]]
#     if len(map_set) == 0:
#         raise ValueError("No bins for mappability correction.")
#     select_idx = np.random.choice(map_set, min(len(map_set), sample_size), replace=False)

#     map_sel = gc_df.loc[select_idx, "MAP"].to_numpy()
#     cor_sel = gc_df.loc[select_idx, "cor_gc"].to_numpy()
#     low_map_fit = lowess(cor_sel, map_sel, frac=0.3, return_sorted=True)
#     f_map = interp1d(low_map_fit[:, 0], low_map_fit[:, 1],
#                     bounds_error=False, fill_value="extrapolate")
#     gc_df["cor_map"] = gc_df["cor_gc"] / np.clip(f_map(gc_df["MAP"]), 1e-8, None)
#     gc_df.loc[gc_df["cor_map"] <= 0, "cor_map"] = np.nan
#     return gc_df
