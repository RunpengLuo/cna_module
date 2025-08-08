import pandas as pd
import numpy as np

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
    meta_refs = refs * gts[:, None] + alts * (1 - gts[:, None])
    meta_ref = np.sum(meta_refs, axis=0)
    meta_alt = np.sum(tots, axis=0) - meta_ref
    return meta_ref, meta_alt

def annotate_meta_snps(
    snp_info: pd.DataFrame,
    ref_mat: np.ndarray,
    alt_mat: np.ndarray,
    dp_mat: np.ndarray,
    nsamples: int,
    max_snps_per_block: int,
    max_blocksize: int,
):
    """
    each region is divided into meta-SNP bins
    assign each SNP with a meta-SNP id
    """
    print("form meta SNPs")

    snp_gts = snp_info["GT"].to_numpy()
    snp_pss = snp_info["PS"].to_numpy()
    snp_unphased = pd.isna(snp_gts)
    snp_gts[pd.isna(snp_gts)] = 1
    snp_gts = snp_gts.astype(np.int8)
    snp_bss = snp_info["Blocksize"].to_numpy()

    snp_info["meta_id"] = -1
    meta_id = 0
    snp_info_grps = snp_info.groupby(by="region_id", sort=False)
    for region_id in snp_info["region_id"].unique():
        region_snps = snp_info_grps.get_group(region_id)
        nsnp_bin = len(region_snps)
        snp_bin_idx = region_snps.index.to_numpy()
        pss = snp_pss[snp_bin_idx]
        unphased = snp_unphased[snp_bin_idx]
        bss = snp_bss[snp_bin_idx]

        meta_ids = np.zeros(nsnp_bin, dtype=np.int16)

        prev_start = 0
        prev_num_snps = 0
        prev_blocksize = 0
        prev_ps = -1
        for i in range(nsnp_bin):
            ps, bs = pss[i], bss[i]
            if unphased[i]:
                # clean previous meta-SNP
                if prev_num_snps > 0:
                    meta_ids[prev_start:i] = meta_id
                    meta_id += 1
                # add current unphased SNP
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
            meta_ids[prev_start:] = meta_id
            meta_id += 1
        snp_info.loc[snp_bin_idx, "meta_id"] = meta_ids
    
    num_meta_snps = meta_id
    print(f"#raw-SNPs={len(snp_info)}\t#meta-SNPs={num_meta_snps}")

    meta_refs = np.zeros((num_meta_snps, nsamples), dtype=np.int32)
    meta_alts = np.zeros((num_meta_snps, nsamples), dtype=np.int32)
    
    snp_info_metas = snp_info.groupby(by="meta_id", sort=False)
    for meta_id in range(meta_ids):
        meta_idx = snp_info_metas.get_group(meta_id).index.to_numpy()
        meta_ref, meta_alt = form_meta_snp(ref_mat[meta_idx, :], alt_mat[meta_idx, :], dp_mat[meta_idx, :], snp_gts[meta_idx])
        meta_refs[:, meta_id] = meta_ref
        meta_alts[:, meta_id] = meta_alt

    return snp_info, meta_refs, meta_alts

def form_meta_info(snp_info: pd.DataFrame):
    """
    form meta-SNP snp info, retain fields #CHR START END #SNPS
    """
    assert "meta_id" in snp_info.columns
    snp_info_metas = snp_info.groupby(by="meta_id", sort=False, as_index=True)
    meta_info = snp_info_metas.agg(
        **{
            "#CHR": ("#CHR", "first"),
            "START": ("START", "min"),
            "END": ("END", "max"),
        }
    ).sort_index().reset_index(drop=True)
    meta_info.loc[:, "#SNPS"] = snp_info_metas.size().reset_index(drop=True)
    return meta_info


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

    # TODO normalization should be done based on allelic balanced bins
    print("RDR library normalization")
    total_bases_normal = np.sum(bases_mat[:, 0])
    total_bases_tumors = np.sum(bases_mat[:, 1:], axis=0)
    correction = total_bases_normal / total_bases_tumors
    print(f"RDR normalization factor: {correction}")
    rdr_mat_corr = rdr_mat_raw * correction[:, None]
    return rdr_mat_raw, rdr_mat_corr
