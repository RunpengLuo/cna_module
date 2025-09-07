import numpy as np
import pandas as pd

def compute_site_switch_error(
    snp_cis: np.ndarray, 
    snp_trans: np.ndarray, 
    snp_tots: np.ndarray, 
    snp_gts2d: np.ndarray,
):
    """ snp_gts[:,0] indicates left-GT for SNP or Meta-SNP"""
    nsnp = len(snp_gts2d)
    site_switch_counts = np.zeros(nsnp, dtype=np.int32)
    for i in range(1, nsnp):
        if snp_tots[i] == 0:
            continue
        gt0, gt1 = snp_gts2d[i - 1, 1], snp_gts2d[i, 0]
        site_switch_counts[i] = snp_trans[i] if gt0 == gt1 else snp_cis[i]
    
    site_switch_errors = np.divide(site_switch_counts, snp_tots, 
                                   where=snp_tots > 0, 
                                   out=np.ones(nsnp) * 0.5)
    return site_switch_errors, site_switch_counts
