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

##################################################
def build_haplo_blocks(
    snp_info: pd.DataFrame,
    ref_mat: np.ndarray,
    alt_mat: np.ndarray,
    tot_mat: np.ndarray,
    nsamples: int,
    min_snp_covering_reads=200,
    min_snp_per_block=10,
    colname="HB"
):
    """
    adaptive binning per phaseset, if a block fails MSR condition, 
    it will be absorbed by adjacent blocks, and its SNP info is ignored.
    """
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
            acc_read_count = tot_mat[ps_snps_index[0], 1:].copy() # tumor sample only
            acc_num_snp = 1
            for i in range(1, nfeatures):
                idx = ps_snps_index[i]
                if (np.all(acc_read_count >= min_snp_covering_reads) 
                    and acc_num_snp >= min_snp_per_block):
                    hb_ids[prev_start:i] = hb_id
                    hb_id += 1
                    prev_start = i

                    acc_read_count = tot_mat[idx, 1:].copy()
                    acc_num_snp= 1
                else:
                    # extend feature block
                    acc_read_count += tot_mat[idx, 1:]
                    acc_num_snp += 1

            # fill last block if any
            hb_ids[prev_start:] = max(hb_id - 1, hb_id0)
            snp_info.loc[ps_snps_index, colname] = hb_ids
            
            # if only a partial block is found, block_id is not incremented in the loop
            # we do it here in this case
            hb_id = max(hb_id, hb_id0 + 1)
    
    haplo_snps_blocks = snp_info.groupby(by=colname, sort=False, as_index=True)
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
    haplo_blocks[colname] = haplo_blocks.index

    raw_phase = snp_info["PHASE_RAW"].to_numpy()
    raw_b_allele_mat = ref_mat * raw_phase[:, None] + alt_mat * (1 - raw_phase[:, None])

    num_blocks = len(haplo_blocks)
    t_allele_mat = np.zeros((num_blocks, nsamples), dtype=np.int32)
    b_allele_mat = np.zeros((num_blocks, nsamples), dtype=np.int32)
    a_allele_mat = np.zeros((num_blocks, nsamples), dtype=np.int32)
    for block_id, block_snps in snp_info.groupby(by=colname, sort=False):
        b_allele_mat[block_id] = np.sum(raw_b_allele_mat[block_snps.index.to_numpy(), :], axis=0)
        t_allele_mat[block_id] = np.sum(tot_mat[block_snps.index.to_numpy(), :], axis=0)
    a_allele_mat = t_allele_mat - b_allele_mat

    print(f"#haplotype-blocks={num_blocks}")
    print(f"min #SNPs per block: ", np.min(haplo_blocks["#SNPS"]))
    print(f"median #SNPs per block: ", np.median(haplo_blocks["#SNPS"]))
    print(f"max #SNPs per block: ", np.max(haplo_blocks["#SNPS"]))
    print(f"min #SNP-covering-reads: ", np.min(t_allele_mat, axis=0))
    print(f"median #SNP-covering-reads: ", np.median(t_allele_mat, axis=0))
    print(f"max #SNP-covering-reads: ", np.max(t_allele_mat, axis=0))
    print(f"median blocksize: ", np.median(haplo_blocks["BLOCKSIZE"]))

    hb_below_msr = np.any(t_allele_mat[:, 1:] < min_snp_covering_reads, axis=1)
    print(f"#blocks failed MSR condition: {np.sum(hb_below_msr)}, {np.sum(hb_below_msr)/num_blocks:.3%}")

    # filter MSR failed blocks
    haplo_blocks_pass: pd.DataFrame = haplo_blocks.loc[~hb_below_msr, :].copy(deep=True)
    t_allele_mat = t_allele_mat[~hb_below_msr]
    a_allele_mat = a_allele_mat[~hb_below_msr]
    b_allele_mat = b_allele_mat[~hb_below_msr]

    # absorb bounderies
    haplo_blocks_pass = fill_gap_haplo_block(haplo_blocks_pass)

    print(f"#haplotype-blocks (after filter&fill)={len(haplo_blocks_pass)}")
    print(f"min #SNPs per block: ", np.min(haplo_blocks_pass["#SNPS"]))
    print(f"median #SNPs per block: ", np.median(haplo_blocks_pass["#SNPS"]))
    print(f"max #SNPs per block: ", np.max(haplo_blocks_pass["#SNPS"]))
    print(f"min #SNP-covering-reads: ", np.min(t_allele_mat, axis=0))
    print(f"median #SNP-covering-reads: ", np.median(t_allele_mat, axis=0))
    print(f"max #SNP-covering-reads: ", np.max(t_allele_mat, axis=0))
    print(f"median blocksize: ", np.median(haplo_blocks_pass["BLOCKSIZE"]))
    return snp_info, haplo_blocks_pass, a_allele_mat, b_allele_mat, t_allele_mat

def fill_gap_haplo_block(haplo_blocks: pd.DataFrame):
    """
        absorb bounderies within segment
    """
    blk_regs = haplo_blocks.groupby(by="region_id", sort=False)
    for region_id in haplo_blocks["region_id"].unique():
        region_blks_raw = blk_regs.get_group(region_id)
        region_blks = blk_regs.get_group(region_id)
        region_idxs = region_blks.index

        # fill left
        if region_blks_raw["START"].iloc[0] < region_blks["START"].iloc[0]:
            haplo_blocks.at[region_idxs[0], "START"] = region_blks_raw["START"].iloc[0]

        for j in range(1, len(region_blks)):
            prev_end = region_blks.loc[region_idxs[j - 1], "END"]
            curr_start = region_blks.loc[region_idxs[j], "START"]
            if prev_end == curr_start:
                continue
            # found a gap, fill it
            mid = int((prev_end + curr_start) // 2)
            haplo_blocks.at[region_idxs[j - 1], "END"] = mid
            haplo_blocks.at[region_idxs[j], "START"] = mid
    haplo_blocks["BLOCKSIZE"] = haplo_blocks["END"] - haplo_blocks["START"]
    return haplo_blocks
