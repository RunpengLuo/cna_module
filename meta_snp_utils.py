import pandas as pd
import numpy as np
from scipy.stats import norm

def binom_proportion_test(beta_mat: np.ndarray, tot_mat: np.ndarray, idx, jdx, alpha=0.1, tol=10e-6):
    xi = beta_mat[idx]
    xj = beta_mat[jdx]
    ni, nj = tot_mat[idx], tot_mat[jdx]
    pi, pj = xi / ni, xj / nj
    pi = np.clip(pi, tol, 1-tol)
    pj = np.clip(pj, tol, 1-tol)
    p = np.clip((xi + xj) / (ni + nj), tol, 1-tol)
    denom = np.sqrt(p * (1 - p) * (1 / ni + 1 / nj))
    z = (pi - pj) / denom
    z_cdf = norm.cdf(z)
    pvals = np.choose(z > 0, [2 * z_cdf, 2 * (1 - z_cdf)])
    return np.all(pvals < alpha)

def form_meta_snp(
    refs: np.ndarray, 
    alts: np.ndarray, 
    tots: np.ndarray,
    gts: np.ndarray
):
    """
    refs, alts, tots: (nsamples, nsnps)
    gts: (nsnps, )
    """
    meta_refs = np.choose(gts[:, None], [alts, refs])
    meta_alts = tots - meta_refs
    meta_ref = np.sum(meta_refs, axis=0)
    meta_alt = np.sum(meta_alts, axis=0)
    return meta_ref, meta_alt

def annotate_meta_snps(
    snp_info: pd.DataFrame,
    ref_mat: np.ndarray,
    alt_mat: np.ndarray,
    tot_mat: np.ndarray,
    site_switch_errors: np.ndarray,
    nsamples: int,
    max_snps_per_block=10,
    max_blocksize=25000,
    alpha=0.1,
    switch_err_rate=0.10,
    colname="meta_id"
):
    """
    form meta-SNPs, bounded by trans ratio, binom test, PS tag
    """
    print("annotate meta-SNPs")
    snp_gts = snp_info["GT"].to_numpy().astype(np.int8)
    snp_pss = snp_info["PS"].to_numpy()
    snp_bss = snp_info["Blocksize"].to_numpy()

    beta_mat = np.zeros_like(ref_mat)
    for si in range(nsamples):
        beta_mat[:, si] = np.choose(snp_gts, [alt_mat[:, si], ref_mat[:, si]])

    snp_info[colname] = -1
    meta_id = 0
    snp_info_grps = snp_info.groupby(by="region_id", sort=False)
    for region_id in snp_info["region_id"].unique():
        region_snps = snp_info_grps.get_group(region_id)
        nsnp_reg = len(region_snps)
        snp_idx = region_snps.index.to_numpy()
        meta_ids = np.zeros(nsnp_reg, dtype=np.uint32)
        prev_start = 0
        prev_num_snps = 1
        prev_idx = snp_idx[0]
        prev_blocksize = snp_bss[prev_idx]
        for i in range(1, nsnp_reg):
            idx = snp_idx[i]
            # pass_binom = binom_proportion_test(beta_mat[:, 1:], tot_mat[:, 1:], prev_idx, idx, alpha=alpha)
            pass_binom = True
            if (
                prev_num_snps >= max_snps_per_block
                or prev_blocksize >= max_blocksize
                or (site_switch_errors[idx] > switch_err_rate)
                or (snp_pss[prev_idx] != snp_pss[idx])
                or not pass_binom
            ):
                meta_ids[prev_start:i] = meta_id
                meta_id += 1
                prev_start = i
                prev_num_snps = 1
                prev_blocksize = snp_bss[idx]
                prev_idx = idx
            else:  # extend meta-SNP
                prev_num_snps += 1
                prev_blocksize += snp_bss[idx]

        if prev_num_snps > 0: # clean final meta-SNP
            meta_ids[prev_start:] = meta_id
            meta_id += 1
        snp_info.loc[snp_idx, colname] = meta_ids
        # print(nsnp_reg, len(np.unique(meta_ids)))
    num_meta_snps = meta_id
    meta_refs = np.zeros((num_meta_snps, nsamples), dtype=np.int32)
    meta_alts = np.zeros((num_meta_snps, nsamples), dtype=np.int32)
    
    snp_info_metas = snp_info.groupby(by=colname, sort=False)
    for mi in range(num_meta_snps):
        meta_idx = snp_info_metas.get_group(mi).index.to_numpy()
        meta_ref, meta_alt = form_meta_snp(ref_mat[meta_idx, :], alt_mat[meta_idx, :], 
                                           tot_mat[meta_idx, :], snp_gts[meta_idx])
        meta_refs[mi, :] = meta_ref
        meta_alts[mi, :] = meta_alt
        # meta_refs and meta_alts are phased within PS block.
    return snp_info, meta_refs, meta_alts

def form_meta_info(snp_info: pd.DataFrame):
    """
    form meta-SNP snp info, retain fields
    """
    assert "meta_id" in snp_info.columns
    snp_info_metas = snp_info.groupby(by="meta_id", sort=False, as_index=True)
    meta_info = snp_info_metas.agg(
        **{
            "#CHR": ("#CHR", "first"),
            "START": ("START", "min"),
            "END": ("END", "max"),
            "PS": ("PS", "first"),
            "region_id": ("region_id", "first")
        }
    ).sort_index().reset_index(drop=True)
    meta_info.loc[:, "#SNPS"] = snp_info_metas.size().reset_index(drop=True)
    return meta_info
