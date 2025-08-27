import os
import sys
import time
import numpy as np
import pandas as pd

from utils import *
from combine_counts_utils import *
from plot_utils import plot_1d2d

if __name__ == "__main__":
    ##################################################
    args = sys.argv
    print(args)
    _, allele_dir, phase_dir, out_dir = sys.argv[:4]
    msr = int(sys.argv[4])

    # input files
    sample_file = os.path.join(allele_dir, "sample_ids.tsv")
    snp_ifile = os.path.join(allele_dir, "snp_info.tsv.gz")
    red_mfile = os.path.join(allele_dir, "snp_matrix.dp.npz")
    ref_mfile = os.path.join(allele_dir, "snp_matrix.ref.npz")
    alt_mfile = os.path.join(allele_dir, "snp_matrix.alt.npz")

    phase_file = os.path.join(phase_dir, "phased.vcf.gz")
    nhair_file = os.path.join(phase_dir, "Normal_hairs.tsv.gz")
    thair_file = os.path.join(phase_dir, "Tumor_hairs.tsv.gz")

    # output files
    os.makedirs(out_dir, exist_ok=True)
    out_cov_mat = os.path.join(out_dir, "bin_matrix.cov.npz")
    out_potts_state_mat = os.path.join(out_dir, "bin_matrix.potts.npz")
    out_baf_raw_mat = os.path.join(out_dir, "bin_matrix.baf_raw.npz")
    out_baf_mat = os.path.join(out_dir, "bin_matrix.baf.npz")
    out_rdr_raw_mat = os.path.join(out_dir, "bin_matrix.rdr_raw.npz")
    out_rdr_mat = os.path.join(out_dir, "bin_matrix.rdr.npz")
    out_bin_file = os.path.join(out_dir, "bin_info.tsv.gz")
    out_snp_file = os.path.join(out_dir, "bin_snps.tsv.gz")

    out_bb_file = os.path.join(out_dir, "bulk.bb") # TODO remove it!

    ##################################################
    print("load arguments")

    samples = pd.read_table(sample_file, sep="\t").loc[:, "SAMPLE"].tolist()
    normal, tumor_samples = samples[0], samples[1:]
    ntumor_samples = len(tumor_samples)
    nsamples = len(samples)

    snp_info = pd.read_table(snp_ifile, sep="\t")
    dp_mat = np.load(red_mfile)["mat"].astype(np.int32)
    base_mat = dp_mat * snp_info["Blocksize"].to_numpy()[:, None]

    ref_mat = np.load(ref_mfile)["mat"].astype(np.int32)
    alt_mat = np.load(alt_mfile)["mat"].astype(np.int32)
    tot_mat = (ref_mat + alt_mat).astype(np.int32)

    phased_vcf = read_VCF(phase_file, phased=True)
    snp_info["_order"] = range(len(snp_info))
    snp_info = pd.merge(left=snp_info, right=phased_vcf, on=["#CHR", "POS"], how="left")
    snp_info = snp_info.sort_values("_order")

    nhairs = load_hairs(nhair_file, smoothing=False)
    thairs = load_hairs(thair_file, smoothing=False)
    hairs = nhairs + thairs
    assert len(hairs) == len(snp_info)

    ##################################################
    snp_info = adaptive_binning(snp_info, tot_mat, msr)
    bin_ids = snp_info["bin_id"].unique()
    nbins = len(bin_ids)
    print(f"#bins={nbins}")

    ##################################################
    cov_mat, mix_baf_mat, mix_icl_mat, nomix_baf_mat, nomix_icl_mat = compute_BAF(
        bin_ids, snp_info, ref_mat, alt_mat, tot_mat, hairs, nbins, nsamples, mirror_mhBAF=True, v=1
    )

    potts_state_mat, potts_baf_mat = baf_model_select(
        bin_ids, snp_info, mix_baf_mat, mix_icl_mat, nomix_baf_mat, nomix_icl_mat, nbins
    )
    baf_mat = potts_baf_mat

    np.savez_compressed(out_cov_mat, mat=cov_mat)
    np.savez_compressed(out_baf_raw_mat, mat=mix_baf_mat)
    np.savez_compressed(out_baf_mat, mat=baf_mat)
    np.savez_compressed(out_potts_state_mat, mat=potts_state_mat)

    ##################################################
    rdr_mat_raw, rdr_mat = compute_RDR(bin_ids, snp_info, base_mat, nbins, ntumor_samples)

    np.savez_compressed(out_rdr_raw_mat, mat=rdr_mat_raw)
    np.savez_compressed(out_rdr_mat, mat=rdr_mat)

    ##################################################
    print("save updated snp_info")
    print(snp_info.columns)
    snp_info.to_csv(
        out_snp_file,
        sep="\t",
        header=True,
        index=False,
        columns=["#CHR", "START", "END", "POS", "Blocksize", "region_id", "bin_id", "PHASE"],
    )

    print("save bin_info")
    snp_grp_bins = snp_info.groupby(by="bin_id", sort=False, as_index=True)

    snp_bins = snp_grp_bins.agg(
        **{
            "#CHR": ("#CHR", "first"),
            "START": ("START", "min"),
            "END": ("END", "max"),
            "region_id": ("region_id", "first")
        }
    )
    snp_bins.loc[:, "#SNP"] = snp_grp_bins.size().reset_index(drop=True)
    snp_bins.to_csv(
        out_bin_file,
        sep="\t",
        header=True,
        index=False,
        columns=["#CHR", "START", "END", "#SNP"],
    )

    bb_df = pd.DataFrame(
        {
            "#CHR": np.repeat(snp_bins["#CHR"], ntumor_samples),
            "START": np.repeat(snp_bins["START"], ntumor_samples),
            "END": np.repeat(snp_bins["END"], ntumor_samples),
            "SAMPLE": np.tile(tumor_samples, nbins),
            "#SNPS": np.repeat(snp_bins["#SNP"], ntumor_samples),
        }
    )

    feature_mats = [
        rdr_mat,
        baf_mat[:, 1:],
        cov_mat[:, 1:],
        rdr_mat_raw,
    ]
    feature_names = ["RD", "BAF", "COV", "UNCORR_RD"]
    for name, mat in zip(feature_names, feature_mats):
        bb_df[name] = mat.flatten()

    # convert to 1-bed?
    bb_df.to_csv(out_bb_file, sep="\t", header=True, index=False)

    # plot RDR and BAF scatter
    plot_1d2d(
        out_dir,
        out_dir,
        out_prefix="",
        plot_normal=True,
        clusters=None,
        expected_rdrs=None,
        expected_bafs=None,
        plot_potts=True
    )
    sys.exit(0)
