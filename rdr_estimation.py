import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pybedtools import BedTool

import matplotlib.pyplot as plt
import seaborn as sns

def compute_gc_content(bin_info: pd.DataFrame, ref_file: str):
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
    return gc_df

def compute_RDR(
    bin_ids: np.ndarray,
    bin_info: pd.DataFrame,
    snp_info: pd.DataFrame,
    bases_mat: np.ndarray,
    nbins: int,
    ntumor_samples: int,
    correct_gc=True,
    ref_file=None,
    out_dir=None
):
    print("compute RDR")

    total_bases_normal = np.sum(bases_mat[:, 0])
    total_bases_tumors = np.sum(bases_mat[:, 1:], axis=0)
    library_correction = total_bases_normal / total_bases_tumors
    print(f"RDR library normalization factor: {library_correction}")

    snp_grp_bins = snp_info.groupby(by="bin_id", sort=False)
    bin_bases_mat = np.zeros((nbins, 1 + ntumor_samples), dtype=np.int64)
    for bin_id in bin_ids:
        snp_bin = snp_grp_bins.get_group(bin_id)
        snp_bin_idx = snp_bin.index.to_numpy()
        bin_bases_mat[bin_id, :] = np.sum(bases_mat[snp_bin_idx], axis=0)

    bin_bss = (bin_info["END"] - bin_info["START"]).to_numpy()

    # read-depth normalized by bin length
    bin_depth_mat = bin_bases_mat / bin_bss[:, None]

    raw_rdr_mat = bin_depth_mat[:, 1:] / bin_depth_mat[:, 0][:, None]
    raw_rdr_mat = raw_rdr_mat * library_correction[:, None]
    if correct_gc:
        print("correct for GC biases")
        gc_df = compute_gc_content(bin_info, ref_file)
        gc = gc_df["GC"].to_numpy()

        # gccorr_dp_mat = np.zeros_like(bin_depth_mat, dtype=np.float64)
        # for si in range(ntumor_samples + 1):
        #     sample_depth = bin_depth_mat[:, si]
        #     df = pd.DataFrame({"RD": sample_depth, "GC": gc})
        #     mod = smf.quantreg("RD ~ GC + I(GC**2)", data=df).fit(q=0.5)
        #     df["GCCORR"] = mod.predict(df)
        #     corr_rdrs = df["RD"] / df["GCCORR"].where(
        #         (df["GCCORR"] > 0) & ~pd.isnull(df["GCCORR"]), 1
        #     )
        #     gccorr_dp_mat[:, si] = corr_rdrs / np.mean(corr_rdrs)
        #     fig, axes = plt.subplots(1, 2, figsize=(8,4))
        #     sns.scatterplot(x=gc, y=sample_depth, ax=axes[0])
        #     sns.scatterplot(x=gc, y=gccorr_dp_mat[:, si], ax=axes[1])
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(out_dir, f"gc{si}.png"), dpi=300)
        #     fig.clear()
        #     plt.close()
        # gccorr_rdr_mat = gccorr_dp_mat[:, 1:] / gccorr_dp_mat[:, 0][:, None]

        gccorr_rdr_mat = np.zeros_like(raw_rdr_mat, dtype=np.float64)
        for si in range(ntumor_samples):
            sample_rdr = raw_rdr_mat[:, si]
            # gc-correction via median quantile regression
            df = pd.DataFrame({"RD": sample_rdr, "GC": gc})
            mod = smf.quantreg("RD ~ GC + I(GC**2)", data=df).fit(q=0.5)
            df["GCCORR"] = mod.predict(df)
            corr_rdrs = df["RD"] / df["GCCORR"].where(
                (df["GCCORR"] > 0) & ~pd.isnull(df["GCCORR"]), 1
            )
            gccorr_rdr_mat[:, si] = corr_rdrs / np.mean(corr_rdrs)
        raw_rdr_mat = gccorr_rdr_mat
    rdr_mat = raw_rdr_mat

    # region_ids = bin_info["region_id"].unique()
    # reg_bins = bin_info.groupby(by="region_id", sort=False)
    # for region_id in region_ids:
    #     reg_bin = reg_bins.get_group(region_id)
    #     reg_rds = rdr_mat[reg_bin.index.to_numpy(), :]
    #     ch = reg_bin["#CHR"].iloc[0]
    #     print(f"{ch}\tmean-RD={np.mean(reg_rds, axis=0)}\tstd-RD={np.std(reg_rds, axis=0)}")

    return rdr_mat

def correct_gc_biases(bin_df: pd.DataFrame, rdr_mat: np.ndarray, ref_file: str):
    print("correct GC biases")
    gc_df = bin_df.merge(
        BedTool.from_dataframe(bin_df[["#CHR", "START", "END"]])
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

    gccorr_rdr_mat = np.zeros_like(rdr_mat, dtype=np.float64)
    gc_df["RD"] = 0.0
    gc_df["GCCORR"] = 0.0
    for si in range(rdr_mat.shape[1]):
        gc_df["RD"] = rdr_mat[:, si]
        mod = smf.quantreg("RD ~ GC + I(GC ** 2.0)", data=gc_df).fit(q=0.5)
        gc_df["GCCORR"] = mod.predict(gc_df[["GC"]])
        corr_rdrs = gc_df["RD"] / gc_df["GCCORR"].where(
            (gc_df["GCCORR"] > 0) & ~pd.isnull(gc_df["GCCORR"]), 1
        )
        gccorr_rdr_mat[:, si] = corr_rdrs / np.mean(corr_rdrs)
    
    return gccorr_rdr_mat
