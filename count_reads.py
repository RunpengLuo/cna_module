import os
import sys

import numpy as np
import pandas as pd

from utils import *

def assign_snp_bounderies(snp_positions: pd.DataFrame, regions: pd.DataFrame):
    """
    divide regions into subregions, each subregion has one SNP
    """
    print("assign SNP bounderies")
    snp_info: pd.DataFrame = snp_positions.copy(deep=True)
    snp_info["POS0"] = snp_info["POS"] - 1
    snp_info["START"] = 0
    snp_info["END"] = 0

    snp_info["region_id"] = 0
    region_id = 0

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

            # annotate region ID            
            snp_info.loc[reg_snp_indices, "region_id"] = region_id
            region_id += 1

            # build SNP bounderies
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
    
    snp_info["Blocksize"] = snp_info["END"] - snp_info["START"]
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

"""
1. Form REF/ALT matrices, (snp, sample), and sample orders
2. Form SNP bounderies, run mosdepth
3. Return ref_mat, alt_mat, dp_mat
"""
if __name__ == "__main__":
    args = sys.argv
    print(args)
    _, region_file, baf_dir, vcf_file, out_dir = args[:5]
    threads = int(args[5])
    readquality = int(args[6])
    bams = args[7:]

    mosdepth = "mosdepth"

    nbaf_file = os.path.join(baf_dir, "normal.1bed")
    tbaf_file = os.path.join(baf_dir, "tumor.1bed")

    os.makedirs(out_dir, exist_ok=True)
    out_dp_file = os.path.join(out_dir, "snp_matrix.dp.npz")
    out_ref_mat = os.path.join(out_dir, "snp_matrix.ref.npz")
    out_alt_mat = os.path.join(out_dir, "snp_matrix.alt.npz")
    out_sid_file = os.path.join(out_dir, "sample_ids.tsv")

    # with boundary information
    out_snp_file = os.path.join(out_dir, "snp_info.tsv.gz")
    out_bed_file = os.path.join(out_dir, "snp_positions.bed.gz")

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

    # exclude any SNPs not found in VCF file
    # this VCF file is served as a white-list
    wl_snps = read_VCF(vcf_file, phased=True)
    all_snps["_order"] = range(len(all_snps))
    all_snps = pd.merge(left=all_snps, right=wl_snps, on=["#CHR", "POS"], how="left")
    all_snps = all_snps.sort_values("_order").drop(columns="_order")

    # filtering
    all_snps = all_snps[~pd.isna(all_snps["GT"]), :].reset_index(drop=True)

    pivot_snps = all_snps.pivot(
        index=["#CHR", "POS"], columns="SAMPLE", values=["REF", "ALT"]
    )
    num_valid_snps = len(pivot_snps.index)
    print(f"#valid Het-SNPs={num_valid_snps}")
    assert num_valid_snps > 0, "no valid HET SNPs after filtering"

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
        columns=["#CHR", "START", "END", "POS", "Blocksize", "region_id"],
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
