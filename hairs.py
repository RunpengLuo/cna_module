import os
import sys
import gzip
import numpy as np
import subprocess
from io import StringIO
import pandas as pd

"""
hair: one read per row, list of phasing supported
n=#SNPs
hair array: shape (n, 4)
"""

def read_hair(hair_file: str, nsnps: int):
    hairs = np.zeros((nsnps, 4), dtype=np.int16)
    mapper = {"00": 0, "01": 1, "10": 2, "11": 3}
    ctr = 0
    with open(hair_file, "r") as hair_fd:
        for line in hair_fd:
            ctr += 1
            if ctr % 100000 == 0:
                print(ctr)
            fields = line.strip().split(' ')[:-1]
            nblocks = int(fields[0])
            for i in range(nblocks):
                var_start = int(fields[2 + (i * 2)]) # 1-based
                phases = fields[2 + (i * 2 + 1)]
                var_end = var_start + len(phases) - 1

                pvar_idx = [mapper[phases[j:j+2]] for j in range(len(phases) - 1)]
                # print(line)
                # print(pvar_idx)
                # print(np.arange(var_start, var_end))
                # print(var_start, var_end, phases)
                hairs[np.arange(var_start, var_end), pvar_idx] += 1
    print(f"total processed {ctr} reads")
    return hairs

def read_VCF(vcf_file: str, has_phase=False):
    if has_phase:
        fields = "%CHROM\t%POS\t[%GT\t%PS]\n"
        names = ["CHR", "POS", "GT", "PS"]
    else:
        fields = "%CHROM\t%POS\n"
        names = ["CHR", "POS"]
    cmd = ["bcftools", "query", "-f", fields, vcf_file]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    snps = pd.read_csv(StringIO(result.stdout), sep="\t", header=None, names=names)
    if has_phase:
        # Drop entries without phasing output
        snps = snps[(snps["GT"] != ".") & (snps["PS"] != ".")]
        snps["GT"] = snps["GT"].apply(func=lambda v: v[0])  # USEREF
    assert not snps.duplicated().any(), f"{vcf_file} has duplicated rows"
    assert not snps.duplicated(subset=["CHR", "POS"]).any(), (
        f"{vcf_file} has duplicated rows"
    )
    return snps

_, hair_file, vcf_file, out_file = sys.argv

nsnps = len(read_VCF(vcf_file))
hairs = read_hair(hair_file, nsnps)

with gzip.open(out_file, "wt") as f:
    np.savetxt(f, hairs, fmt="%d", delimiter="\t")
    f.close()
