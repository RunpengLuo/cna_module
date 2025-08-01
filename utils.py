import subprocess
from io import StringIO

import pandas as pd
import numpy as np
import pysam


def get_ord2chr(ch="chr"):
    return [f"{ch}{i}" for i in range(1, 23)] + [f"{ch}X", f"{ch}Y"]


def get_chr2ord(ch):
    chr2ord = {}
    for i in range(1, 23):
        chr2ord[f"{ch}{i}"] = i
    chr2ord[f"{ch}X"] = 23
    chr2ord[f"{ch}Y"] = 24
    return chr2ord


def sort_chroms(chromosomes: list):
    assert len(chromosomes) != 0
    chromosomes = [str(c) for c in chromosomes]
    ch = "chr" if str(chromosomes[0]).startswith("chr") else ""
    chr2ord = get_chr2ord(ch)
    return sorted(chromosomes, key=lambda x: chr2ord[x])


def read_baf_file(baf_file: str):
    baf_df = pd.read_table(
        baf_file,
        names=["#CHR", "POS", "SAMPLE", "REF", "ALT"],
        dtype={
            "#CHR": object,
            "POS": np.uint32,
            "SAMPLE": object,
            "REF": np.uint32,
            "ALT": np.uint32,
        },
    )
    return baf_df


def subset_baf(
    baf_df: pd.DataFrame, ch: str, start: int, end: int, is_last_block=False
):
    if ch != None:
        baf_ch = baf_df[baf_df["#CHR"] == ch]
    else:
        baf_ch = baf_df
    if baf_ch.index.name == "POS":
        pos = baf_ch.index
    else:
        pos = baf_ch["POS"]
    if is_last_block:
        return baf_ch[(pos >= start) & (pos <= end)]
    else:
        return baf_ch[(pos >= start) & (pos < end)]


def read_VCF(vcf_file: str, phased=False):
    """
    load vcf file as dataframe.
    If phased, parse GT[0] as USEREF, check PS
    """
    fields = "%CHROM\t%POS"
    names = ["#CHR", "POS"]
    format_tags = []
    if phased:
        vcf = pysam.VariantFile(vcf_file)
        format_tags.extend(["%GT"])
        names.extend(["GT"])
        if "PS" in vcf.header.formats:
            format_tags.extend(["%PS"])
            names.extend(["PS"])
        vcf.close()
        fields = fields + "\t[" + "\t".join(format_tags) + "]"
    fields += "\n"
    cmd = ["bcftools", "query", "-f", fields, vcf_file]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    snps = pd.read_csv(StringIO(result.stdout), sep="\t", header=None, names=names)
    assert not snps.duplicated().any(), f"{vcf_file} has duplicated rows"
    assert not snps.duplicated(subset=["#CHR", "POS"]).any(), (
        f"{vcf_file} has duplicated rows"
    )
    if phased:
        # Drop entries without phasing output
        if "PS" not in snps.columns:
            snps.loc[:, "PS"] = 0
        snps = snps[(~snps["GT"].isna()) & snps["GT"].str.contains(r"\|", na=False)]
        snps["GT"] = snps["GT"].apply(func=lambda v: v[0])
        snps.loc[:, "GT"] = snps["GT"].astype("Int8")
        snps.loc[~snps["GT"].isna(), "GT"] = (
            1 - snps.loc[~snps["GT"].isna(), "GT"]
        )  # USEREF
        snps.loc[:, "PS"] = snps["PS"].astype("Int64")
    snps = snps.reset_index(drop=True)
    return snps
