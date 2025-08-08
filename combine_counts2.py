import os
import sys
import time
import numpy as np
import pandas as pd

from utils import *
from combine_counts_utils import *
from phase_snps import *
from plot_utils import plot_1d2d
from statsmodels.stats.multitest import multipletests


if __name__ == "__main__":
    ##################################################
    args = sys.argv
    print(args)
    _, phase_file, rdr_dir, out_dir = args[:4]
    msr, mtr, read_length, max_snps_per_block, max_blocksize = list(map(int, args[4:]))
    mrd = mtr * read_length

    sample_file = os.path.join(rdr_dir, "sample_ids.tsv")
    snp_ifile = os.path.join(rdr_dir, "snp_info.tsv.gz")
    red_mfile = os.path.join(rdr_dir, "snp_matrix.dp.npz")
    ref_mfile = os.path.join(rdr_dir, "snp_matrix.ref.npz")
    alt_mfile = os.path.join(rdr_dir, "snp_matrix.alt.npz")

    samples = pd.read_table(sample_file, sep="\t").loc[:, "SAMPLE"].tolist()
    normal = samples[0]
    tumor_samples = samples[1:]
    ntumor_samples = len(tumor_samples)
    nsamples = len(samples)

    os.makedirs(out_dir, exist_ok=True)
    out_cov_mat = os.path.join(out_dir, "bin_matrix.cov.npz")
    out_baf_mat = os.path.join(out_dir, "bin_matrix.baf.npz")
    out_rdr_raw_mat = os.path.join(out_dir, "bin_matrix.rdr_raw.npz")
    out_rdr_corr_mat = os.path.join(out_dir, "bin_matrix.rdr_corr.npz")
    out_alpha_mat = os.path.join(out_dir, "bin_matrix.alpha.npz")
    out_beta_mat = os.path.join(out_dir, "bin_matrix.beta.npz")
    out_bb_file = os.path.join(out_dir, "bulk.bb")
    out_bin_file = os.path.join(out_dir, "bin_position.tsv.gz")
    out_snp_file = os.path.join(out_dir, "phased_snps.tsv.gz")

    out_meta_ifile = os.path.join(out_dir, "meta_info.tsv.gz")
    out_meta_ref_mat = os.path.join(out_dir, "meta_matrix.ref.npz")
    out_meta_alt_mat = os.path.join(out_dir, "meta_matrix.alt.npz")

    ##################################################
    print("load arguments")

    snp_info = pd.read_table(snp_ifile, sep="\t")
    red_mat = np.load(red_mfile)["mat"].astype(np.int32)
    ref_mat = np.load(ref_mfile)["mat"].astype(np.int32)
    alt_mat = np.load(alt_mfile)["mat"].astype(np.int32)
    dp_mat = (ref_mat + alt_mat).astype(np.int32)

    phased_vcf = read_VCF(phase_file, phased=True)

    chroms = snp_info["#CHR"].unique()

    snp_info["_order"] = range(len(snp_info))
    snp_info = pd.merge(left=snp_info, right=phased_vcf, on=["#CHR", "POS"], how="left")
    snp_info = snp_info.sort_values("_order")

    # assign region id for consecutive SNPs
    same_chr = snp_info["#CHR"] == snp_info["#CHR"].shift()
    adj_pos = snp_info["START"] == snp_info["END"].shift()
    new_cluster = ~(same_chr & adj_pos)
    snp_info["region_id"] = new_cluster.cumsum()

    snp_info["Blocksize"] = snp_info["END"] - snp_info["START"]
    bases_mat = red_mat * snp_info["Blocksize"].to_numpy()[:, None]

    ##################################################
    # build meta-SNPs
    snp_info, meta_refs, meta_alts = annotate_meta_snps(snp_info, ref_mat, alt_mat,
                                                        dp_mat, nsamples, 
                                                        max_snps_per_block, max_blocksize)

    meta_info = form_meta_info(snp_info)

    np.savez_compressed(out_meta_ref_mat, mat=meta_refs)
    np.savez_compressed(out_meta_alt_mat, mat=meta_alts)

    meta_info.to_csv(
        out_meta_ifile,
        sep="\t",
        header=True,
        index=False,
    )
    sys.exit(0)
