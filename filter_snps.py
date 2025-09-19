import os
import sys

import numpy as np
import pandas as pd
import pyranges as pr
from scipy.stats import beta

from utils import *

def validate_snps(pivot_snps: pd.DataFrame, regions: pd.DataFrame):
    """
    filter any SNP doesn't belong to valid regions
    filter any SNP doesn't appear in all SAMPLE
    """
    print("exclude SNPs not in region file")
    all_positions = pivot_snps.index.to_frame(index=False)
    all_positions["POS0"] = all_positions["POS"] - 1
    all_positions["End"] = all_positions["POS0"] + 1

    pr_snps = pr.PyRanges(
        all_positions.rename(columns={"#CHR": "Chromosome", "POS0": "Start"})
    )
    pr_regions = pr.PyRanges(
        regions.rename(columns={"#CHR": "Chromosome", "START": "Start", "END": "End"})
    )
    overlapping_snps = pr_snps.overlap(pr_regions).df
    overlapping_snps = overlapping_snps.rename(columns={"Chromosome": "#CHR"})
    mask = (
        pd.merge(
            left=all_positions,
            right=overlapping_snps,
            on=["#CHR", "POS"],
            how="left",
            sort=False,
        )["Start"]
        .isna()
        .to_numpy()
    )
    pivot_snps = pivot_snps.loc[~mask, :]
    pivot_snps = pivot_snps.dropna(how="any")
    print(f"#SNP after validation={len(pivot_snps)}")
    return pivot_snps

def filter_snps(snp_positions: pd.DataFrame, ref_mat: np.ndarray, alt_mat: np.ndarray, min_ad: int, gamma: float):
    """
    exclude SNPs have 0 count in any of the SAMPLE
    exclude SNPs if normal sample failed beta-posterior credible interval test with beta(1, 1) prior.
    """
    print("filter SNPs")
    snp_wl = np.ones(len(snp_positions), dtype=bool)
    snp_wl = snp_wl & np.all((ref_mat + alt_mat) >= min_ad, axis=1)

    p_lower = gamma / 2.0
    p_upper = 1.0 - p_lower
    q = np.array([p_lower, p_upper])
    het_cred_ints = beta.ppf(q[None, :], ref_mat[:, 0][:, None] + 1, alt_mat[:, 0][:, None] + 1)
    het_incl_balanced = (het_cred_ints[:, 0] <= 0.5) & (0.5 <= het_cred_ints[:, 1])
    snp_wl = snp_wl & het_incl_balanced

    snp_positions = snp_positions.loc[snp_wl, :]
    ref_mat = ref_mat[snp_wl, :]
    alt_mat = alt_mat[snp_wl, :]
    print(f"#SNP after filtering={len(snp_positions)}")
    return snp_positions, ref_mat, alt_mat

"""
1. Filter Het SNPs based on certain criteria, 
2. return valid SNP positions (snps.1pos) to filter VCF file.
"""
if __name__ == "__main__":
    args = sys.argv
    print(args)
    _, region_file, baf_dir, out_1pos_file = args[:4]
    min_ad = int(args[4])
    gamma = float(args[5])

    nbaf_file = os.path.join(baf_dir, "normal.1bed")
    tbaf_file = os.path.join(baf_dir, "tumor.1bed")

    ##################################################
    print("load arguments")
    regions = pd.read_table(
        region_file,
        sep="\t",
        header=None,
        usecols=range(3),
        names=["#CHR", "START", "END"],
    )

    normal_snps = read_baf_file(nbaf_file)
    tumor_snps = read_baf_file(tbaf_file)
    samples = normal_snps["SAMPLE"].unique().tolist() + tumor_snps["SAMPLE"].unique().tolist()
    print("samples=", samples)

    all_snps = pd.concat([normal_snps, tumor_snps], axis=0).reset_index(drop=True)
    all_snps["#CHR"] = pd.Categorical(
        all_snps["#CHR"], categories=get_ord2chr(ch="chr"), ordered=True
    )
    all_snps.sort_values(by=["#CHR", "POS"], inplace=True, ignore_index=True)

    pivot_snps = all_snps.pivot(
        index=["#CHR", "POS"], columns="SAMPLE", values=["REF", "ALT"]
    )
    print(f"#SNPs (raw)={len(pivot_snps.index)}")
    pivot_snps = validate_snps(pivot_snps, regions)

    ##################################################
    # preserve sample order
    pivot_samples = pivot_snps.columns.levels[1].tolist()
    sample2i = {sample: i for i, sample in enumerate(samples)}
    pivot_orders = np.argsort([sample2i[sample] for sample in pivot_samples])

    snp_positions: pd.DataFrame = pivot_snps.index.to_frame(index=False)
    ref_mat = pivot_snps["REF"].to_numpy()[:, pivot_orders]
    alt_mat = pivot_snps["ALT"].to_numpy()[:, pivot_orders]

    snp_positions, ref_mat, alt_mat = filter_snps(snp_positions, ref_mat, alt_mat, min_ad, gamma)
    assert len(snp_positions) > 0, "no valid HET SNPs after filtering"

    snp_positions.to_csv(
        out_1pos_file,
        columns=["#CHR", "POS"],
        sep="\t",
        header=False,
        index=False
    )
    sys.exit(0)
