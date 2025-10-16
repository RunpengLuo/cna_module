import numpy as np
import pandas as pd

def compute_site_switch_TGS(
    snp_cis: np.ndarray, 
    snp_trans: np.ndarray, 
    snp_tots: np.ndarray, 
    snp_gts2d: np.ndarray,
):
    """
    snp_gts2d[:,0] indicates left-GT for SNP or Meta-SNP
    Output:
    1. site_switch_counts, col1=#reads supporting current phase, and col2=#reads supporting switched phase.
    2. site_switch_errors: sse[i] denotes the switch error or switch probability for site i-1 and i.
    """
    nsnp = len(snp_gts2d)
    site_switch_counts = np.zeros((nsnp, 2), dtype=np.int32)
    for i in range(1, nsnp):
        if snp_tots[i] == 0:
            continue
        gt0, gt1 = snp_gts2d[i - 1, 1], snp_gts2d[i, 0]
        if gt0 == gt1:
            site_switch_counts[i, 0] = snp_cis[i]
            site_switch_counts[i, 1] = snp_trans[i]
        else:
            site_switch_counts[i, 0] = snp_trans[i]
            site_switch_counts[i, 1] = snp_cis[i]
    
    site_switch_errors = np.divide(site_switch_counts[:, 1], snp_tots, 
                                   where=snp_tots > 0, 
                                   out=np.ones(nsnp) * 0.5)
    return site_switch_errors, site_switch_counts


def compute_site_switch_error_NGS():
    pass
