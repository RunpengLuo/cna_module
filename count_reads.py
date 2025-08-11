import os
import sys
import subprocess

import pyranges as pr

import numpy as np
import pandas as pd
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

def filter_snps(snp_positions: pd.DataFrame, ref_mat: np.ndarray, alt_mat: np.ndarray, gamma: float):
    """
    exclude SNPs have 0 count in any of the SAMPLE
    exclude SNPs if normal sample failed beta-posterior credible interval test.
    """
    print("filter SNPs")
    snp_wl = np.ones(len(snp_positions), dtype=bool)
    snp_wl = snp_wl & np.all((ref_mat + alt_mat) > 0, axis=1)

    def isHet(countA, countB, gamma):
        p_lower = gamma / 2.0
        p_upper = 1.0 - p_lower
        [c_lower, c_upper] = beta.ppf([p_lower, p_upper], countA + 1, countB + 1)
        return c_lower <= 0.5 <= c_upper
    snp_wl = snp_wl & np.vectorize(isHet)(ref_mat[:, 0], alt_mat[:, 0], gamma)

    snp_positions = snp_positions.loc[snp_wl, :]
    ref_mat = ref_mat[snp_wl, :]
    alt_mat = alt_mat[snp_wl, :]
    print(f"#SNP after filtering={len(snp_positions)}")
    return snp_positions, ref_mat, alt_mat

def assign_snp_bounderies(snp_positions: pd.DataFrame, regions: pd.DataFrame):
    """
    divide regions into subregions, each subregion has one SNP
    """
    print("assign SNP bounderies")
    snp_info: pd.DataFrame = snp_positions.copy(deep=True)
    snp_info["POS0"] = snp_info["POS"] - 1
    snp_info["START"] = 0
    snp_info["END"] = 0

    chroms = snp_info["#CHR"].unique().tolist()
    region_grps_ch = regions.groupby(by="#CHR", sort=False)
    for chrom in chroms:
        regions_ch = region_grps_ch.get_group(chrom)
        for _, region in regions_ch.iterrows():
            reg_start, reg_end = region["START"], region["END"]
            reg_snps = subset_baf(snp_info, chrom, reg_start, reg_end)
            if len(reg_snps) == 0:
                continue
            reg_snp_positions = reg_snps["POS0"].to_numpy()
            reg_snp_indices = reg_snps.index.to_numpy()
            if len(reg_snps) == 1:
                snp_info.loc[reg_snp_indices, "START"] = reg_start
                snp_info.loc[reg_snp_indices, "END"] = reg_end
            else:
                reg_bounderies = np.ceil(
                    np.vstack([reg_snp_positions[:-1], reg_snp_positions[1:]]).mean(
                        axis=0
                    )
                ).astype(np.uint32)
                reg_bounderies = np.concatenate(
                    [[reg_start], reg_bounderies, [reg_end]]
                )
                snp_info.loc[reg_snp_indices, "START"] = reg_bounderies[:-1]
                snp_info.loc[reg_snp_indices, "END"] = reg_bounderies[1:]
    return snp_info


def run_mosdepth(
    samples: list,
    bams: list,
    threads: int,
    readquality: int,
    snp_bed_file: str,
    mosdepth_odir: str,
):
    print("run mosdepth to compute per region depth")
    mos_files = []
    for sample, bam in zip(samples, bams):
        mos_file = f"{mosdepth_odir}/{sample}.regions.bed.gz"
        if not os.path.exists(mos_file):
            print(f"run mosdepth on {sample}")
            msdp_cmd = [
                mosdepth,
                "--no-per-base",
                "--fast-mode",
                "-t",
                str(threads),
                "-Q",
                str(readquality),
                "--by",
                snp_bed_file,
                f"{mosdepth_odir}/{sample}",
                bam,
            ]
            err_fd = open(f"{mosdepth_odir}/{sample}.err.log", "w")
            out_fd = open(f"{mosdepth_odir}/{sample}.out.log", "w")
            ret = subprocess.run(msdp_cmd, stdout=out_fd, stderr=err_fd)
            err_fd.close()
            out_fd.close()
            ret.check_returncode()
        assert os.path.exists(mos_file)
        mos_files.append(mos_file)
    return mos_files


if __name__ == "__main__":
    ##################################################
    args = sys.argv
    print(args)
    _, sample, region_file, baf_dir, out_dir = args[:5]
    bams = args[5:]
    assert len(bams) == 2
    samples = ["normal", sample]

    nbaf_file = os.path.join(baf_dir, "normal.1bed")
    tbaf_file = os.path.join(baf_dir, "tumor.1bed")

    os.makedirs(out_dir, exist_ok=True)
    out_dp_file = os.path.join(out_dir, "snp_matrix.dp.npz")
    out_ref_mat = os.path.join(out_dir, "snp_matrix.ref.npz")
    out_alt_mat = os.path.join(out_dir, "snp_matrix.alt.npz")
    out_sid_file = os.path.join(out_dir, "sample_ids.tsv")
    out_snp_file = os.path.join(out_dir, "snp_info.tsv.gz")
    out_bed_file = os.path.join(out_dir, "snp_positions.bed.gz")

    mosdepth = "mosdepth"
    threads = 8
    readquality = 11
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

    pd.DataFrame({"SAMPLE": samples}).to_csv(
        out_sid_file, sep="\t", header=True, index=False
    )
    snp_positions: pd.DataFrame = pivot_snps.index.to_frame(index=False)
    ref_mat = pivot_snps["REF"].to_numpy()[:, pivot_orders]
    alt_mat = pivot_snps["ALT"].to_numpy()[:, pivot_orders]

    snp_positions, ref_mat, alt_mat = filter_snps(snp_positions, ref_mat, alt_mat, gamma=0.05) # TODO
    assert len(snp_positions) > 0, "no valid HET SNPs after filtering"

    np.savez_compressed(out_ref_mat, mat=ref_mat)
    np.savez_compressed(out_alt_mat, mat=alt_mat)

    ##################################################
    snp_info = assign_snp_bounderies(snp_positions, regions)
    snp_info.to_csv(
        out_snp_file,
        sep="\t",
        header=True,
        index=False,
        columns=["#CHR", "START", "END", "POS"],
    )
    snp_info.to_csv(
        out_bed_file,
        columns=["#CHR", "START", "END"],
        sep="\t",
        header=False,
        index=False,
    )

    ##################################################
    mosdepth_odir = os.path.join(out_dir, f"out_mosdepth")
    os.makedirs(mosdepth_odir, exist_ok=True)
    mos_files = run_mosdepth(
        samples, bams, threads, readquality, out_bed_file, mosdepth_odir
    )

    snp_info["_order"] = range(len(snp_info))
    for sample, mos_file in zip(samples, mos_files):
        mos_df = pd.read_table(
            mos_file, sep="\t", header=None, names=["#CHR", "START", "END", "COV"]
        )
        snp_info = pd.merge(
            left=snp_info, right=mos_df, how="left", on=["#CHR", "START", "END"]
        )
        snp_info = snp_info.sort_values("_order")
        snp_info = snp_info.rename(columns={"COV": sample})

    dp_mat = snp_info.loc[:, samples].to_numpy()
    np.savez_compressed(out_dp_file, mat=dp_mat)
    sys.exit(0)
