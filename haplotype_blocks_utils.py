import pandas as pd
import numpy as np


from utils import *

##################################################
def load_genetic_map(genetic_map_file: str, mode="eagle2", sep=" "):
    assert mode == "eagle2"
    genetic_map = pd.read_table(genetic_map_file, sep=sep, index_col=None).rename(
        columns={"chr": "#CHR", 
                 "position": "POS", 
                 "COMBINED_rate(cM/Mb)": "recomb_rate",
                 "Genetic_Map(cM)": "pos_cm"}
    )
    genetic_map["#CHR"] = genetic_map["#CHR"].astype(str)
    genetic_map.loc[genetic_map["#CHR"] == "23", "#CHR"] = "X"
    if not genetic_map["#CHR"].str.startswith("chr").any():
        genetic_map["#CHR"] = "chr" + genetic_map["#CHR"].astype(str)
    genetic_map = sort_df_chr(genetic_map, ch="#CHR", pos="POS")
    genetic_map["POS0"] = genetic_map["POS"] - 1
    return genetic_map

##################################################
def estimate_switchprob_genetic_map(
    snp_info: pd.DataFrame,
    genetic_map_file: str,
    nu=1,
    min_switchprob=1e-6
):
    """
    compute switchprobs for adjacent SNPs or bins.
    For bins, the bin boundary is used.
    """
    print("compute prior phase-switch probability from genetic map")
    genetic_map = load_genetic_map(genetic_map_file, mode="eagle2")    
    snp_info["d_morgans"] = 0.0
    genetic_map_chrs = genetic_map.groupby(by="#CHR", sort=False, observed=True)
    for ch, ch_snps in snp_info.groupby(by="#CHR", sort=False):
        ch_maps = genetic_map_chrs.get_group(ch)
        start_cms = np.interp(
            ch_snps["START"].to_numpy(),
            ch_maps["POS"].to_numpy(),
            ch_maps["pos_cm"].to_numpy()
        )
        pos_cms = np.interp(
            ch_snps["POS"].to_numpy(),
            ch_maps["POS"].to_numpy(),
            ch_maps["pos_cm"].to_numpy()
        )
        end_cms = np.interp(
            ch_snps["END"].to_numpy(),
            ch_maps["POS"].to_numpy(),
            ch_maps["pos_cm"].to_numpy()
        )

        d_morgans_midpoint = np.zeros(len(pos_cms), dtype=np.float32)
        d_morgans_midpoint[1:] = pos_cms[1:] - pos_cms[:-1]

        d_morgans = np.zeros(len(pos_cms), dtype=np.float32)
        d_morgans[1:] = start_cms[1:] - end_cms[:-1]

        # avoid direct overlapping bins
        d_morgans[d_morgans <= 0] = d_morgans_midpoint[d_morgans <= 0]
        d_morgans = np.maximum(d_morgans, 0.0) # avoid non-monotonic noise if any
        snp_info.loc[ch_snps.index, "d_morgans"] = d_morgans / 100

    
    snp_info["switchprobs"] = 0.0 # P(0->1 or 1->0), lower value favors phase switch
    for ch, ch_snps in snp_info.groupby(by="#CHR", sort=False):
        d_morgans = ch_snps["d_morgans"].to_numpy()
        switchprobs = (1 - np.exp(-2 * nu * d_morgans)) / 2.0
        switchprobs = np.clip(switchprobs, a_min=min_switchprob, a_max=0.5)
        snp_info.loc[ch_snps.index, "switchprobs"] = switchprobs
    
    switchprobs = snp_info["switchprobs"]
    print(f"min-site-switch-error={np.min(switchprobs):.5f}")
    print(f"average-site-switch-error={np.mean(switchprobs):.5f}")
    print(f"median-site-switch-error={np.median(switchprobs):.5f}")
    print(f"max-site-switch-error={np.max(switchprobs):.5f}")
    return snp_info

def estimate_switchprob_long_read(
    snp_info: pd.DataFrame,
    hair_files: list,
    min_switchprob=1e-6
):
    nsnp = len(snp_info)
    snp_gts = snp_info["GT"].to_numpy().astype(np.int8)
    snp_gts2d = np.concatenate([snp_gts[:, None], snp_gts[:, None]], axis=1)

    hairs = np.sum([load_hairs(hair_file) for hair_file in hair_files], axis=0)
    assert len(hairs) == nsnp
    hairs_total = np.sum(hairs, axis=1)

    snp_cis = np.sum(hairs[:, [0, 3]], axis=1)  # 00 11
    snp_trans = np.sum(hairs[:, [1, 2]], axis=1)  # 01 10

    site_switch_counts = np.zeros((nsnp, 2), dtype=np.int32)
    for i in range(1, nsnp):
        if hairs_total[i] == 0:
            continue
        gt0, gt1 = snp_gts2d[i - 1, 1], snp_gts2d[i, 0]
        if gt0 == gt1:
            site_switch_counts[i, 0] = snp_cis[i]
            site_switch_counts[i, 1] = snp_trans[i]
        else:
            site_switch_counts[i, 0] = snp_trans[i]
            site_switch_counts[i, 1] = snp_cis[i]
    
    site_switch_errors = np.divide(site_switch_counts[:, 1], hairs_total, 
                                   where=hairs_total > 0, 
                                   out=np.ones(nsnp) * 0.5)
    print(f"min-site-switch-error={np.min(site_switch_errors):.5f}")
    print(f"average-site-switch-error={np.mean(site_switch_errors):.5f}")
    print(f"median-site-switch-error={np.median(site_switch_errors):.5f}")
    print(f"max-site-switch-error={np.max(site_switch_errors):.5f}")
    
    snp_info["switchprobs"] = np.clip(site_switch_errors, a_min=min_switchprob, a_max=0.5)
    return snp_info

# def compute_site_switch_TGS(
#     snp_cis: np.ndarray, 
#     snp_trans: np.ndarray, 
#     snp_tots: np.ndarray, 
#     snp_gts2d: np.ndarray,
# ):
#     """
#     snp_gts2d[:,0] indicates left-GT for SNP or Meta-SNP
#     Output:
#     1. site_switch_counts, col1=#reads supporting current phase, and col2=#reads supporting switched phase.
#     2. site_switch_errors: sse[i] denotes the switch error or switch probability for site i-1 and i.
#     """
#     nsnp = len(snp_gts2d)
#     site_switch_counts = np.zeros((nsnp, 2), dtype=np.int32)
#     for i in range(1, nsnp):
#         if snp_tots[i] == 0:
#             continue
#         gt0, gt1 = snp_gts2d[i - 1, 1], snp_gts2d[i, 0]
#         if gt0 == gt1:
#             site_switch_counts[i, 0] = snp_cis[i]
#             site_switch_counts[i, 1] = snp_trans[i]
#         else:
#             site_switch_counts[i, 0] = snp_trans[i]
#             site_switch_counts[i, 1] = snp_cis[i]
    
#     site_switch_errors = np.divide(site_switch_counts[:, 1], snp_tots, 
#                                    where=snp_tots > 0, 
#                                    out=np.ones(nsnp) * 0.5)
#     return site_switch_errors, site_switch_counts

##################################################
def build_haplo_blocks(
    snp_info: pd.DataFrame,
    ref_mat: np.ndarray,
    alt_mat: np.ndarray,
    tot_mat: np.ndarray,
    nsamples: int,
    max_snps_per_block=10,
    max_switch_err_rate=0.10,
    colname="HB"
):
    # haplotype block id
    hb_id = 0
    snp_info[colname] = 0
    snp_info["BAF"] = 0.0

    snp_info_regs = snp_info.groupby(by="region_id", sort=False)
    for region_id in snp_info["region_id"].unique():
        region_snps = snp_info_regs.get_group(region_id)
        for ps in region_snps["PS"].unique():
            ps_snps = region_snps.loc[region_snps["PS"] == ps, :]
            ps_snps_index = ps_snps.index
            nfeatures = len(ps_snps)
            hb_ids = np.zeros(nfeatures, dtype=np.int64)

            hb_id0 = hb_id
            prev_start = 0
            prev_num_snps = 1
            for i in range(1, nfeatures):
                idx = ps_snps_index[i]
                if (
                    prev_num_snps < max_snps_per_block
                    and snp_info.loc[idx, "switchprobs"] < max_switch_err_rate
                ):
                    # extend feature block
                    prev_num_snps += 1
                else:
                    hb_ids[prev_start:i] = hb_id
                    hb_id += 1
                    prev_start = i
                    prev_num_snps = 1
            # fill last block if any
            hb_ids[prev_start:] = max(hb_id - 1, hb_id0)
            snp_info.loc[ps_snps_index, colname] = hb_ids
            
            # if only a partial block is found, block_id is not incremented in the loop
            # we do it here in this case
            hb_id = max(hb_id, hb_id0 + 1)
    
    haplo_snps_blocks = snp_info.groupby(by="HB", sort=False, as_index=True)
    haplo_blocks = haplo_snps_blocks.agg(
        **{
            "#CHR": ("#CHR", "first"),
            "START": ("START", "min"),
            "END": ("END", "max"),
            "PS": ("PS", "first"),
            "switchprobs": ("switchprobs", "first"),
            "region_id": ("region_id", "first")
        }
    )
    haplo_blocks.loc[:, "#SNPS"] = haplo_snps_blocks.size().reset_index(drop=True)
    haplo_blocks.loc[:, "BLOCKSIZE"] = haplo_blocks["END"] - haplo_blocks["START"]
    haplo_blocks["HB"] = haplo_blocks.index

    num_blocks = len(haplo_blocks)
    print(f"#haplotype-blocks={num_blocks}")
    print(f"min #SNPS: ", np.min(haplo_blocks["#SNPS"]))
    print(f"max #SNPS: ", np.max(haplo_blocks["#SNPS"]))
    print(f"median #SNPs per block: ", np.median(haplo_blocks["#SNPS"]))
    print(f"median blocksize: ", np.median(haplo_blocks["BLOCKSIZE"]))

    raw_phase = snp_info["PHASE_RAW"].to_numpy()
    raw_b_allele_mat = ref_mat * raw_phase[:, None] + alt_mat * (1 - raw_phase[:, None])

    t_allele_mat = np.zeros((num_blocks, nsamples), dtype=np.int32)
    b_allele_mat = np.zeros((num_blocks, nsamples), dtype=np.int32)
    a_allele_mat = np.zeros((num_blocks, nsamples), dtype=np.int32)
    for block_id, block_snps in snp_info.groupby(by="HB", sort=False):
        b_allele_mat[block_id] = np.sum(raw_b_allele_mat[block_snps.index.to_numpy(), :], axis=0)
        t_allele_mat[block_id] = np.sum(tot_mat[block_snps.index.to_numpy(), :], axis=0)
    a_allele_mat = t_allele_mat - b_allele_mat
    
    return snp_info, haplo_blocks, a_allele_mat, b_allele_mat, t_allele_mat
