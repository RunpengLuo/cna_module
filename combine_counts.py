import os
import sys
import time
import numpy as np
import pandas as pd

from utils import *
from phase_utils import *
from meta_snp_utils import *
from plot_utils import plot_1d2d


from baf_estimation import *
from rdr_estimation import *


# TODO, extend to multi-PS case, and then do phasing internally.
# handle RDR
def adaptive_binning(
    snp_info: pd.DataFrame,
    tot_mat: np.ndarray,
    site_switch_errors: np.ndarray,
    min_snp_reads=3000,
    min_total_bases=10e6,
    switch_err_rate=0.3,  # min_site_switch_error, set to 1.0 to disable
    ps_aware=True,  # restrict each bin within same PS block or unphased.
    colname="bin_id",
):
    print("adaptive binning")
    snp_pss = snp_info["PS"].to_numpy()

    snp_info[colname] = -1
    bin_id = 0
    acc_snp = np.zeros(tot_mat.shape[1] - 1, dtype=np.int32)
    snp_info_grps = snp_info.groupby(by="region_id", sort=False)
    for region_id in snp_info["region_id"].unique():
        region_snps = snp_info_grps.get_group(region_id)
        nsnp_reg = len(region_snps)
        snp_idx = region_snps.index.to_numpy()
        bin_ids = np.zeros(nsnp_reg, dtype=np.uint32)
        prev_start = 0
        prev_num_snps = 1
        prev_idx = snp_idx[0]
        acc_snp[:] = tot_mat[prev_idx, 1:]
        for i in range(1, nsnp_reg):
            idx = snp_idx[i]
            if (
                np.all(acc_snp >= min_snp_reads)
                or (ps_aware and (snp_pss[prev_idx] != snp_pss[idx]))
                # or (site_switch_errors[idx] > switch_err_rate)
            ):
                bin_ids[prev_start:i] = bin_id
                bin_id += 1
                prev_start = i
                prev_num_snps = 1
                prev_idx = idx
                acc_snp[:] = 0
            else:  # extend binning
                prev_num_snps += 1
                acc_snp[:] += tot_mat[idx, 1:]
        if prev_num_snps > 0:  # clean final bin
            bin_ids[prev_start:] = bin_id
            bin_id += 1
        snp_info.loc[snp_idx, colname] = bin_ids
    return snp_info


if __name__ == "__main__":
    ##################################################
    args = sys.argv
    print(args)
    _, ref_file, allele_dir, phase_dir, out_dir = sys.argv[:5]
    msr = int(sys.argv[5])

    verbose = 1
    read_type = "TGS"
    correct_gc = True
    mirror_mhBAF = False
    map_BAF = True
    phase_mode = "prior"
    assert phase_mode in ["prior", "em", "hmm"]
    print(f"phasing_mode={phase_mode}")

    # input files
    sample_file = os.path.join(allele_dir, "sample_ids.tsv")
    snp_ifile = os.path.join(allele_dir, "snp_info.tsv.gz")
    red_mfile = os.path.join(allele_dir, "snp_matrix.dp.npz")
    ref_mfile = os.path.join(allele_dir, "snp_matrix.ref.npz")
    alt_mfile = os.path.join(allele_dir, "snp_matrix.alt.npz")

    phase_file = os.path.join(phase_dir, "phased.vcf.gz")
    if read_type == "TGS":
        nhair_file = os.path.join(phase_dir, "Normal.hairs.tsv.gz")
        thair_file = os.path.join(phase_dir, "Tumor.hairs.tsv.gz")

    # output files
    os.makedirs(out_dir, exist_ok=True)
    out_bin_file = os.path.join(out_dir, "bin_info.tsv.gz")
    out_snp_file = os.path.join(out_dir, "bin_snps.tsv.gz")
    out_bb_file = os.path.join(out_dir, "bulk.bb")  # TODO remove it!

    out_cov_mat = os.path.join(out_dir, "bin_matrix.cov.npz")
    out_rdr_mat = os.path.join(out_dir, "bin_matrix.rdr.npz")
    out_baf_mat = os.path.join(out_dir, "bin_matrix.baf.npz")
    out_alpha_mat = os.path.join(out_dir, "bin_matrix.alpha.npz")
    out_beta_mat = os.path.join(out_dir, "bin_matrix.beta.npz")
    ##################################################
    print("load arguments")
    samples = pd.read_table(sample_file, sep="\t").loc[:, "SAMPLE"].tolist()
    normal, tumor_samples = samples[0], samples[1:]
    ntumor_samples = len(tumor_samples)
    nsamples = len(samples)

    snp_info = pd.read_table(snp_ifile, sep="\t")
    dp_mat = np.load(red_mfile)["mat"].astype(np.int32)
    ref_mat = np.load(ref_mfile)["mat"].astype(np.int32)
    alt_mat = np.load(alt_mfile)["mat"].astype(np.int32)
    tot_mat = (ref_mat + alt_mat).astype(np.int32)

    phased_vcf = read_VCF(phase_file, phased=True)
    snp_info["_order"] = range(len(snp_info))
    snp_info = pd.merge(left=snp_info, right=phased_vcf, on=["#CHR", "POS"], how="left")
    snp_info = snp_info.sort_values("_order")

    # remove unphased if any
    snp_info = snp_info.loc[~pd.isna(snp_info["GT"]), :]  # ignore unphased SNPs
    dp_mat = dp_mat[snp_info.index, :]
    ref_mat = ref_mat[snp_info.index, :]
    alt_mat = alt_mat[snp_info.index, :]
    tot_mat = tot_mat[snp_info.index, :]
    snp_info = snp_info.reset_index(drop=True)
    num_snps = len(snp_info)
    print(f"#SNPs={num_snps}")

    snp_bss = snp_info["Blocksize"].to_numpy()
    bases_mat = dp_mat * snp_bss[:, None]
    snp_pss = snp_info["PS"].to_numpy()
    snp_gts = snp_info["GT"].to_numpy().astype(np.int8)
    snp_gts2d = np.concatenate([snp_gts[:, None], snp_gts[:, None]], axis=1)

    if read_type == "TGS":
        nhairs = load_hairs(nhair_file, smoothing=False)
        thairs = load_hairs(thair_file, smoothing=False)
        hairs = nhairs + thairs
        assert len(hairs) == num_snps
        snp_hairs_total = hairs.sum(axis=1)
        snp_cis = np.sum(hairs[:, [0, 3]], axis=1)  # 00 11
        snp_trans = np.sum(hairs[:, [1, 2]], axis=1)  # 01 10

    ##################################################
    # derive phase-switch errors for adjacent SNPs
    if read_type == "TGS":
        site_switch_errors, site_switch_counts = compute_site_switch_error(
            snp_cis, snp_trans, snp_hairs_total, snp_gts2d
        )
        print(f"average-site-switch-error={np.mean(site_switch_errors):.3f}")
        print(f"median-site-switch-error={np.median(site_switch_errors):.3f}")
    else:
        # NGS TODO
        site_switch_errors = None
        site_switch_counts = None

    snp_info, meta_refs, meta_alts = annotate_meta_snps(
        snp_info,
        ref_mat,
        alt_mat,
        tot_mat,
        site_switch_errors,
        nsamples,
        max_snps_per_block=30,
        max_blocksize=1e5,
        alpha=0.1,
        switch_err_rate=0.1,
        colname="meta_id",
    )
    meta_info = form_meta_info(snp_info)

    meta_tots = meta_refs + meta_alts
    meta_snps = snp_info.groupby("meta_id", sort=False)
    meta_ids = snp_info["meta_id"].unique()
    num_metas = len(meta_ids)
    print(f"#meta-SNPs={num_metas}")

    meta_gts2d = np.zeros((num_metas, 2), dtype=np.int8)
    meta_gts2d[:, 0] = meta_snps["GT"].first().to_numpy()
    meta_gts2d[:, 1] = meta_snps["GT"].last().to_numpy()

    if read_type == "TGS":
        first_idx_metas = meta_snps.apply(lambda x: x.index[0]).to_numpy()
        meta_hairs = hairs[first_idx_metas]
        meta_cis = np.sum(meta_hairs[:, [0, 3]], axis=1)
        meta_trans = np.sum(meta_hairs[:, [1, 2]], axis=1)
        meta_hairs_total = meta_hairs.sum(axis=1).astype(np.int32)
        meta_switch_errors, meta_switch_counts = compute_site_switch_error(
            meta_cis, meta_trans, meta_hairs_total, meta_gts2d
        )
        # switch errors between meta-SNPs
        print(f"average-meta-switch-error={np.mean(meta_switch_errors):.3f}")
        print(f"median-meta-switch-error={np.median(meta_switch_errors):.3f}")
    else:
        pass

    # if verbose > 1:
    # plot meta-SNPs in 1D, skip 2D TODO

    ##################################################
    meta_info = adaptive_binning(
        meta_info,
        meta_tots,
        meta_switch_errors,
        min_snp_reads=msr,
        min_total_bases=10e6,
        switch_err_rate=0.3,
        ps_aware=True,
    )
    bin_ids = meta_info["bin_id"].unique()
    meta_bins = meta_info.groupby("bin_id", sort=False)
    nbins = len(bin_ids)
    print(f"#bins={nbins}")

    # meta_info join snp_info to add bin_id
    meta_info["meta_id"] = meta_info.index.to_numpy()
    snp_info = pd.merge(
        left=snp_info, right=meta_info[["meta_id", "bin_id"]], on="meta_id", how="left"
    )
    snp_info["PHASE"] = 0.0

    # TODO TMP save meta-SNPs information here
    meta_info.to_csv(
        os.path.join(out_dir, "meta_snp_info.tsv.gz"),
        sep="\t",
        header=True,
        index=False,
    )

    np.savez_compressed(
        os.path.join(out_dir, "meta_snp.ref.npz"), mat=meta_refs
    )
    np.savez_compressed(
        os.path.join(out_dir, "meta_snp.alt.npz"), mat=meta_alts
    )

    ##################################################
    if read_type == "TGS":
        if phase_mode == "prior":
            cov_mat, baf_mat, alpha_mat, beta_mat = compute_BAF_prior(
                bin_ids,
                meta_info,
                meta_refs,
                meta_alts,
                meta_tots,
                nbins,
                nsamples,
                mirror_mhBAF=mirror_mhBAF,
                v=verbose,
            )
            np.savez_compressed(out_alpha_mat, mat=alpha_mat)
            np.savez_compressed(out_beta_mat, mat=beta_mat)
        elif phase_mode == "hmm":
            cov_mat, mix_baf_mat, mix_icl_mat, nomix_baf_mat, nomix_icl_mat = (
                compute_BAF(
                    bin_ids,
                    meta_info,
                    meta_refs,
                    meta_alts,
                    meta_tots,
                    meta_hairs,
                    nbins,
                    nsamples,
                    mirror_mhBAF=mirror_mhBAF,
                    map_BAF=map_BAF,
                    v=verbose,
                )
            )
            np.savez_compressed(
                os.path.join(out_dir, "bin_matrix.mix_baf.npz"), mat=mix_baf_mat
            )
            np.savez_compressed(
                os.path.join(out_dir, "bin_matrix.nomix_baf.npz"), mat=nomix_baf_mat
            )
            np.savez_compressed(
                os.path.join(out_dir, "bin_matrix.mix_icl.npz"), mat=mix_icl_mat
            )
            np.savez_compressed(
                os.path.join(out_dir, "bin_matrix.nomix_icl.npz"), mat=nomix_icl_mat
            )

            potts_state_mat, potts_baf_mat, scaled_icl_delta = baf_model_select(
                bin_ids,
                meta_info,
                mix_baf_mat,
                mix_icl_mat,
                nomix_baf_mat,
                nomix_icl_mat,
                nbins,
            )
            np.savez_compressed(
                os.path.join(out_dir, "bin_matrix.potts.npz"), mat=potts_state_mat
            )
            np.savez_compressed(
                os.path.join(out_dir, "bin_matrix.scaled_icl_delta.npz"),
                mat=scaled_icl_delta,
            )
            baf_mat = potts_baf_mat
        else:  # em
            pass
    else:  # NGS
        pass

    np.savez_compressed(out_cov_mat, mat=cov_mat)
    np.savez_compressed(out_baf_mat, mat=baf_mat)
    ##################################################
    snp_grp_bins = snp_info.groupby(by="bin_id", sort=False, as_index=True)

    bin_info = snp_grp_bins.agg(
        **{
            "#CHR": ("#CHR", "first"),
            "START": ("START", "min"),
            "END": ("END", "max"),
            "region_id": ("region_id", "first"),
        }
    )
    bin_info.loc[:, "#SNPS"] = snp_grp_bins.size().reset_index(drop=True)

    ##################################################
    rdr_mat = compute_RDR(
        bin_ids, bin_info, snp_info, bases_mat, nbins, ntumor_samples,
        correct_gc=correct_gc, ref_file=ref_file, out_dir=out_dir
    )

    np.savez_compressed(out_rdr_mat, mat=rdr_mat)

    ##################################################
    print("save snp_info")
    print(snp_info.columns)
    snp_info.to_csv(
        out_snp_file,
        sep="\t",
        header=True,
        index=False,
        columns=[
            "#CHR",
            "START",
            "END",
            "POS",
            "Blocksize",
            "region_id",
            "bin_id",
            "PHASE",
        ],
    )

    print("save bin_info")
    bin_info.to_csv(
        out_bin_file,
        sep="\t",
        header=True,
        index=False,
        columns=["#CHR", "START", "END", "#SNPS", "region_id"],
    )

    bb_df = pd.DataFrame(
        {
            "#CHR": np.repeat(bin_info["#CHR"], ntumor_samples),
            "START": np.repeat(bin_info["START"], ntumor_samples),
            "END": np.repeat(bin_info["END"], ntumor_samples),
            "SAMPLE": np.tile(tumor_samples, nbins),
            "#SNPS": np.repeat(bin_info["#SNPS"], ntumor_samples),
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

    # convert to 1-bed?
    bb_df.to_csv(out_bb_file, sep="\t", header=True, index=False)

    # plot RDR and BAF scatter
    plot_1d2d(
        out_dir,
        out_dir,
        plot_normal=True,
        mirrored_baf=mirror_mhBAF,
        clusters=None,
        expected_rdrs=None,
        expected_bafs=None,
        plot_cov=False,
        plot_alpha_beta=False,
    )
    if phase_mode in ["hmm"]:
        plot_1d2d(
            out_dir,
            out_dir,
            out_prefix=f"{phase_mode}_",
            plot_normal=True,
            mirrored_baf=mirror_mhBAF,
            clusters=None,
            expected_rdrs=None,
            expected_bafs=None,
            plot_potts=True,
            baf_file="bin_matrix.mix_baf.npz",
            rdr_file="bin_matrix.rdr.npz",
            potts_states=potts_state_mat,
        )
    sys.exit(0)
