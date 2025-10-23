import os
import sys
import time
import numpy as np
import pandas as pd

from hatchet_parser import parse_arguments_build_haplotype_blocks

from utils import *
from rdr_estimation import *
from haplotype_blocks_utils import *
from plot_utils_new import plot_1d2d

"""
Build meta-SNPs, output a/b-allele counts and normalized&gc-corrected RDR.
"""
if __name__ == "__main__":
    ##################################################
    args = parse_arguments_build_haplotype_blocks()

    work_dir = args["work_dir"]
    out_dir = os.path.join(work_dir, args["out_dir"])

    read_type = args["read_type"]
    ref_file = args["reference"]
    gmap_file = args["genetic_map"]

    min_snp_covering_reads = int(args["msr"])
    min_snp_per_block = int(args["mspb"])
    max_switch_error = float(args["mserr"])
    correct_gc = not args["no_gc_correct"]

    # input files
    allele_dir = os.path.join(work_dir, "allele")
    sample_file = os.path.join(allele_dir, "sample_ids.tsv")
    snp_ifile = os.path.join(allele_dir, "snp_info.tsv.gz")
    red_mfile = os.path.join(allele_dir, "snp_matrix.dp.npz")
    ref_mfile = os.path.join(allele_dir, "snp_matrix.ref.npz")
    alt_mfile = os.path.join(allele_dir, "snp_matrix.alt.npz")

    phase_dir = os.path.join(work_dir, "phase")
    phase_file = os.path.join(phase_dir, "phased.vcf.gz")

    # output files
    os.makedirs(out_dir, exist_ok=True)
    out_bb_file = os.path.join(out_dir, "bulk.bb")  # TODO remove later
    out_block_file = os.path.join(out_dir, "block_info.tsv.gz")
    out_rdr_mat = os.path.join(out_dir, "block_matrix.rdr.npz")
    out_alpha_mat = os.path.join(out_dir, "block_matrix.alpha.npz")
    out_beta_mat = os.path.join(out_dir, "block_matrix.beta.npz")
    out_total_mat = os.path.join(out_dir, "block_matrix.total.npz")
    out_baf_mat = os.path.join(out_dir, "block_matrix.baf.npz")
    out_cov_mat = os.path.join(out_dir, "block_matrix.cov.npz")

    ##################################################
    print("load arguments")
    samples = pd.read_table(sample_file, sep="\t").loc[:, "SAMPLE"].tolist()
    normal, tumor_samples = samples[0], samples[1:]
    ntumor_samples = len(tumor_samples)
    nsamples = len(samples)

    snp_info = pd.read_table(snp_ifile, sep="\t")
    ref_mat = np.load(ref_mfile)["mat"].astype(np.int32)
    alt_mat = np.load(alt_mfile)["mat"].astype(np.int32)
    tot_mat = (ref_mat + alt_mat).astype(np.int32)

    phased_vcf = read_VCF(phase_file, phased=True)
    snp_info = pd.merge(
        left=snp_info, right=phased_vcf, on=["#CHR", "POS"], how="left", sort=False
    )
    assert np.all(~pd.isna(snp_info["GT"])), (
        "invalid input, only phased SNPs should present here"
    )
    snp_info["PHASE_RAW"] = snp_info["GT"].astype(np.float32)
    num_snps = len(snp_info)
    print(f"#SNPs={num_snps}")

    snp_bss = snp_info["BLOCKSIZE"].to_numpy()
    dp_mat = np.load(red_mfile)["mat"].astype(np.float32)
    bases_mat = dp_mat * snp_bss[:, None]
    # depth is fractional, #bases should be integer.
    bases_mat = np.ceil(bases_mat).astype(np.int64)

    ##################################################
    # derive phase-switch errors for adjacent SNPs by genetic map or PS info
    nu = 1
    min_switchprob = 1e-4
    switch_bias = 1e-4
    snp_info["switchprobs"] = 0.0
    if read_type == "NGS":
        snp_info = estimate_switchprob_genetic_map(snp_info, gmap_file, nu=nu, min_switchprob=min_switchprob)
        # annotate PS info, usually N/A from reference/population-based phasing
        snp_info["PS"] = (snp_info["switchprobs"] > max_switch_error).cumsum()
    else:
        # for long reads, PS information is mostly available from phasing tools
        # use PS information to set switchprobs
        assert np.all(snp_info["PS"].notna()), snp_info.loc[snp_info["PS"].isna(), ::]
        same_block = snp_info["PS"] == snp_info["PS"].shift(1).fillna(False) # first dummy SNP
        snp_info["switchprobs"] = np.where(
            same_block,
            min_switchprob,  # within same phase set
            0.5 - switch_bias    # crossing phase set
        )
    print(snp_info.head())
    num_phasesets = len(snp_info["PS"].unique())
    num_segments = len(snp_info["region_id"].unique())
    print(f"#phasesets={num_phasesets}")
    print(f"#segments={num_segments}")

    ##################################################
    snp_info, haplo_blocks, a_allele_mat, b_allele_mat, t_allele_mat = (
        build_haplo_blocks(
            snp_info,
            ref_mat,
            alt_mat,
            tot_mat,
            nsamples,
            min_snp_covering_reads,
            min_snp_per_block,
            colname="HB",
        )
    )
    block_ids = haplo_blocks["HB"].to_numpy()
    num_blocks = len(block_ids)

    baf_mat = b_allele_mat / t_allele_mat
    cov_mat = t_allele_mat / haplo_blocks["#SNPS"].to_numpy()[:, None]

    haplo_blocks.to_csv(
        out_block_file,
        sep="\t",
        header=True,
        index=False,
        columns=[
            "HB",
            "region_id",
            "#CHR",
            "START",
            "END",
            "BLOCKSIZE",
            "#SNPS",
            "PS",
            "switchprobs",
        ],
    )

    np.savez_compressed(out_alpha_mat, mat=a_allele_mat)
    np.savez_compressed(out_beta_mat, mat=b_allele_mat)
    np.savez_compressed(out_total_mat, mat=t_allele_mat)
    np.savez_compressed(out_baf_mat, mat=baf_mat)
    np.savez_compressed(out_cov_mat, mat=cov_mat)

    ##################################################
    # RDR
    rdr_mat = compute_RDR(
        block_ids,
        haplo_blocks,
        snp_info,
        bases_mat,
        num_blocks,
        ntumor_samples,
        correct_gc=correct_gc,
        ref_file=ref_file,
        out_dir=out_dir,
        grp_id="HB",
    )
    np.savez_compressed(out_rdr_mat, mat=rdr_mat)

    ##################################################
    # plot here TODO

    ##################################################
    # bb file
    bb_df = pd.DataFrame(
        {
            "#CHR": np.repeat(haplo_blocks["#CHR"], ntumor_samples),
            "START": np.repeat(haplo_blocks["START"], ntumor_samples),
            "END": np.repeat(haplo_blocks["END"], ntumor_samples),
            "SAMPLE": np.tile(tumor_samples, num_blocks),
            "#SNPS": np.repeat(haplo_blocks["#SNPS"], ntumor_samples),
        }
    )

    feature_mats = [
        cov_mat[:, 1:],
        baf_mat[:, 1:],
        rdr_mat,
    ]

    feature_names = ["COV", "BAF", "RD"]
    for name, mat in zip(feature_names, feature_mats):
        bb_df[name] = mat.flatten()

    bb_df.to_csv(out_bb_file, sep="\t", header=True, index=False)


    plot_1d2d(
        haplo_blocks,
        baf_mat[:, 1:],
        rdr_mat,
        None,
        None,
        None,
        args["genome_file"],
        out_dir,
        out_prefix="",
        plot_mirror_baf=False,
    )
