import os
import sys
import time
import numpy as np
import pandas as pd

from utils import *
from phase_snps import *
from plot_utils import plot_1d2d
from statsmodels.stats.multitest import multipletests


def adaptive_binning(
    snp_info: pd.DataFrame,
    dp_mat: np.ndarray,
    bases_mat: np.ndarray,
    msr: int,
    mrd: int,
):
    """
    variable-length binning, MSR, MTR
    """
    print("adaptive binning")
    region_ids = snp_info["region_id"].unique()
    snp_info_grps_reg = snp_info.groupby(by="region_id", sort=False)

    snp_info["bin_id"] = -1
    bin_id = 0
    acc_snp = np.zeros(ntumor_samples, dtype=np.int32)
    acc_tot = np.zeros(ntumor_samples, dtype=np.int32)
    for region_id in region_ids:
        acc_snp[:] = 0
        acc_tot[:] = 0
        snp_reg = snp_info_grps_reg.get_group(region_id)
        snp_idx = snp_reg.index.to_numpy()
        start = 0
        for i in range(len(snp_idx)):
            idx = snp_idx[i]
            acc_snp[:] += dp_mat[idx, 1:]
            acc_tot[:] += bases_mat[idx, 1:]
            if np.all(acc_snp >= msr) and np.all(acc_tot >= mrd):
                snp_info.loc[snp_idx[start : i + 1], "bin_id"] = bin_id
                bin_id += 1
                start = i + 1
                acc_snp[:] = 0
                acc_tot[:] = 0

        # handle tail
        if start < len(snp_idx):
            snp_info.loc[snp_idx[start:], "bin_id"] = bin_id
            bin_id += 1
    return snp_info

def meta_snp(
    refs: np.ndarray, 
    alts: np.ndarray, 
    tots: np.ndarray,
    gts: np.ndarray
):
    """
    refs, alts, tots: (nsamples, nsnps)
    gts: (nsnps, )
    """
    meta_refs = refs * gts[:, None] + alts * (1 - gts[:, None])
    meta_ref = np.sum(meta_refs, axis=0)
    meta_alt = np.sum(tots, axis=0) - meta_ref
    return meta_ref, meta_alt


def compute_mhBAF(
    snp_info: pd.DataFrame,
    ref_mat: np.ndarray,
    alt_mat: np.ndarray,
    dp_mat: np.ndarray,
    nbins: int,
    nsamples: int,
    max_snps_per_block: int,
    max_blocksize: int,
):
    """
    TODO easy multithread
    For each bin
    1. form meta-SNP w.r.t. #SNPs, blocksize, phaseset
    2. EM phasing over tumor samples
    3. EM phasing over normal sample
    """
    print("compute phased mhBAF")

    mirror_baf = True
    bin_flips = np.random.randint(2, size=nbins)

    ts = time.time()

    cov_mat = np.zeros((nbins, nsamples), dtype=np.float64)
    baf_mat = np.zeros((nbins, nsamples), dtype=np.float64)
    alpha_mat = np.zeros((nbins, nsamples), dtype=np.int32)
    beta_mat = np.zeros((nbins, nsamples), dtype=np.int32)

    snp_gts = snp_info["GT"].to_numpy()
    snp_pss = snp_info["PS"].to_numpy()
    snp_unphased = pd.isna(snp_gts)
    snp_gts[pd.isna(snp_gts)] = 1
    snp_gts = snp_gts.astype(np.int8)
    snp_bss = snp_info["Blocksize"].to_numpy()
    snp_grp_bins = snp_info.groupby(by="bin_id", sort=False)
    bin_ids = snp_info["bin_id"].unique()

    rbaf_mat = np.zeros((nbins, nsamples), dtype=np.float64)
    raw_pvals = np.zeros((nbins, 2), dtype=np.float64)

    for bin_id in bin_ids:
        if bin_id % 500 == 0:
            print(f"process {bin_id}/{nbins} {time.time() - ts}")
        snp_bin = snp_grp_bins.get_group(bin_id)
        snp_bin_idx = snp_bin.index.to_numpy()
        nsnp_bin = len(snp_bin)
        refs = ref_mat[snp_bin_idx]
        alts = alt_mat[snp_bin_idx]
        tots = dp_mat[snp_bin_idx]
        gts = snp_gts[snp_bin_idx]
        pss = snp_pss[snp_bin_idx]
        unphased = snp_unphased[snp_bin_idx]
        bss = snp_bss[snp_bin_idx]

        # meta-SNP info
        meta_refs_arr = []
        meta_alts_arr = []
        meta_ids = np.zeros(nsnp_bin, dtype=np.int16)

        prev_start = 0
        prev_num_snps = 0
        prev_blocksize = 0
        prev_ps = -1
        meta_id = 0
        for i in range(nsnp_bin):
            ps, bs = pss[i], bss[i]
            if unphased[i]:
                # clean previous meta-SNP
                if prev_num_snps > 0:
                    meta_ref, meta_alt = meta_snp(refs[prev_start:i], alts[prev_start:i], tots[prev_start:i], gts[prev_start:i])
                    meta_refs_arr.append(meta_ref)
                    meta_alts_arr.append(meta_alt)
                    meta_ids[prev_start:i] = meta_id
                    meta_id += 1
                # add current unphased SNP
                meta_refs_arr.append(refs[i])
                meta_alts_arr.append(alts[i])
                meta_ids[i] = meta_id
                meta_id += 1
                prev_start = i + 1
                prev_num_snps = 0
                prev_blocksize = 0
                prev_ps = -1
            else:
                if prev_ps != -1:
                    if (
                        prev_num_snps >= max_snps_per_block
                        or prev_blocksize >= max_blocksize
                        or (prev_ps != ps)
                    ):
                        meta_ref, meta_alt = meta_snp(refs[prev_start:i], alts[prev_start:i], tots[prev_start:i], gts[prev_start:i])
                        meta_refs_arr.append(meta_ref)
                        meta_alts_arr.append(meta_alt)

                        meta_ids[prev_start:i] = meta_id
                        meta_id += 1
                        prev_start = i
                        prev_num_snps = 1
                        prev_blocksize = bs
                        prev_ps = ps
                    else:  # extend meta-SNP
                        prev_num_snps += 1
                        prev_blocksize += bs
                else:  # init meta-SNP
                    prev_start = i
                    prev_num_snps = 1
                    prev_blocksize = bs
                    prev_ps = ps
        if prev_start < nsnp_bin:  # clean final meta-SNP
            meta_ref, meta_alt = meta_snp(refs[prev_start:], alts[prev_start:], tots[prev_start:], gts[prev_start:])
            meta_refs_arr.append(meta_ref)
            meta_alts_arr.append(meta_alt)
    
            meta_ids[prev_start:] = meta_id
            meta_id += 1

        # meta-snps, sample by #SNP
        refs = np.vstack(meta_refs_arr).astype(np.int32).T
        alts = np.vstack(meta_alts_arr).astype(np.int32).T
        totals = refs + alts

        runs = {
            b: multisample_em(alts[1:], refs[1:], b, mirror=mirror_baf) 
            for b in np.arange(0, 0.55, 0.05)
        }
        bafs, phases, ll = max(runs.values(), key=lambda x: x[-1])

        rbafs, rphases, rll = random_phasing(alts[1:], refs[1:], totals[1:])
        rbaf_mat[bin_id, 1:] = rbafs
        raw_pvals[bin_id, 0] = binom_test_approx(refs[0][None, :], totals[0][None, :])
        raw_pvals[bin_id, 1] = binom_test_approx(refs[1:], totals[1:])
        
        
        if not mirror_baf:
            flip_baf = bin_flips[bin_id]
            bafs = flip_baf * (1 - bafs) + (1 - flip_baf) * bafs
            phases = flip_baf * (1 - phases) + (1 - flip_baf) * phases

        snp_info.loc[snp_bin_idx, "PHASE"] = phases[meta_ids] * gts + (
            1 - phases[meta_ids]
        ) * (1 - gts)

        # mean ENTROPY of phasing posterior
        snp_info.loc[snp_bin_idx, "ENTROPY"] = get_phasing_entropy(phases)[meta_ids]

        sample_totals = np.sum(refs + alts, axis=1)
        bAlleles = refs @ phases + alts @ (1 - phases)

        baf_mat[bin_id, 0] = bAlleles[0] / sample_totals[0]
        baf_mat[bin_id, 1:] = bafs

        phases = np.round(phases).astype(np.int8)
        betas = refs @ phases + alts @ (1 - phases)
        alpha_mat[bin_id, :] = sample_totals - betas
        beta_mat[bin_id, :] = betas
        cov_mat[bin_id,] = sample_totals / nsnp_bin
    
    # FDR BH correction
    # pval_mask = ~np.isnan(raw_pvals)
    # rejected, _, _, _ = multipletests(raw_pvals[pval_mask], method="fdr_bh", alpha=0.05)
    # pval_mask[pval_mask] = ~rejected
    # # pval_mask == True if allelic balanced.
    # baf_mat[pval_mask, :] = rbaf_mat[pval_mask, :]

    return cov_mat, baf_mat, alpha_mat, beta_mat, raw_pvals


def compute_RDR(
    bin_ids: np.ndarray, snp_info: pd.DataFrame, bases_mat: np.ndarray, nbins: int, ntumor_samples: int
):
    print("compute RDR")
    snp_grp_bins = snp_info.groupby(by="bin_id", sort=False)
    rdr_mat_raw = np.zeros((nbins, ntumor_samples), dtype=np.float64)
    for bin_id in bin_ids:
        snp_bin = snp_grp_bins.get_group(bin_id)
        snp_bin_idx = snp_bin.index.to_numpy()
        agg_bases_normal = np.sum(bases_mat[snp_bin_idx, 0])
        if agg_bases_normal > 0:
            agg_bases_tumors = np.sum(bases_mat[snp_bin_idx, 1:], axis=0)
            rdr_mat_raw[bin_id, :] = agg_bases_tumors / agg_bases_normal

    print("RDR library normalization")
    total_bases_normal = np.sum(bases_mat[:, 0])
    total_bases_tumors = np.sum(bases_mat[:, 1:], axis=0)
    correction = total_bases_normal / total_bases_tumors
    print(f"RDR normalization factor: {correction}")
    rdr_mat_corr = rdr_mat_raw * correction[:, None]
    return rdr_mat_raw, rdr_mat_corr


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
    snp_info = adaptive_binning(snp_info, dp_mat, bases_mat, msr, mrd)
    bin_ids = snp_info["bin_id"].unique()
    nbins = len(bin_ids)
    print(f"#bins={nbins}")

    ##################################################
    snp_info["PHASE"] = 0.0
    snp_info["ENTROPY"] = 0.0
    cov_mat, baf_mat, alpha_mat, beta_mat, raw_pvals = compute_mhBAF(
        snp_info,
        ref_mat,
        alt_mat,
        dp_mat,
        nbins,
        nsamples,
        max_snps_per_block,
        max_blocksize,
    )

    # bin by sample matrix
    np.savez_compressed(out_cov_mat, mat=cov_mat)
    np.savez_compressed(out_baf_mat, mat=baf_mat)
    np.savez_compressed(out_alpha_mat, mat=alpha_mat)
    np.savez_compressed(out_beta_mat, mat=beta_mat)
    np.savez_compressed(os.path.join(out_dir, "bin_matrix.pval.npz"), mat=raw_pvals)

    ##################################################
    rdr_mat_raw, rdr_mat_corr = compute_RDR(bin_ids, snp_info, bases_mat, nbins, ntumor_samples)

    np.savez_compressed(out_rdr_raw_mat, mat=rdr_mat_raw)
    np.savez_compressed(out_rdr_corr_mat, mat=rdr_mat_corr)

    ##################################################
    print("build bb dataframe")
    # build bb dataframe and compute RDR
    # per-sample bb-row for later steps, maybe improved speed via RD matrix instead.
    snp_info.to_csv(
        out_snp_file,
        sep="\t",
        header=True,
        index=False,
        columns=["#CHR", "START", "END", "POS", "bin_id", "PHASE", "ENTROPY"],
    )

    snp_grp_bins = snp_info.groupby(by="bin_id", sort=False, as_index=True)

    snp_bins = snp_grp_bins.agg(
        **{
            "#CHR": ("#CHR", "first"),
            "START": ("START", "min"),
            "END": ("END", "max"),
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
        rdr_mat_corr,
        baf_mat[:, 1:],
        cov_mat[:, 1:],
        alpha_mat[:, 1:],
        beta_mat[:, 1:],
        rdr_mat_raw,
    ]
    feature_names = ["RD", "BAF", "COV", "ALPHA", "BETA", "UNCORR_RD"]
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
    )
    sys.exit(0)
