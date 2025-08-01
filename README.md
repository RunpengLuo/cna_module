# Long reads CNA calling pipeline (Companion with HATCHet2)

## Limitations
1. currently pipeline works for one sample, but will be adapted to multi-sample pretty soon

## Global Input
1. normal.bam
2. tumor.bam
3. dbSNP candidate SNP .vcf.gz file
4. reference .fa file

## Preprocessing
### Input
1. dbSNP candidate SNP .vcf.gz file

### Steps
1. HET SNPs genotyping on normal sample
2. allele counting on tumor sample against genotyped HET SNPs
3. phase normal sample

### Output
1. `snps/chr*.vcf.gz`
2. `baf/normal.1bed` and `baf/tumor.1bed`
3. `phase/phased.vcf.gz`

## Counting
### Input
1. region.bed
2. `baf/`

### Steps
1. obtain HET SNPs consistently found among all samples
2. mask regions not found in region.bed
3. divide chromosomes within each region, such that each bin has 1 HET SNP
4. bin-level read depth counting

### Output
1. `rdr/sample_ids.tsv`
2. `rdr/snp_info.tsv.gz`
4. `rdr/snp_matrix.ref.npz`
5. `rdr/snp_matrix.alt.npz`
6. `rdr/snp_matrix.dp.npz`

each snp_matrix has shape (bins, samples), with order specified via sample_ids.tsv and snp_info.tsv.gz

## Adaptive Binning
### Input
1. `rdr/`
2. `phase/`
3. MSR and MTR
4. MSPB

### Steps
1. merge adjacent bins into segment, such that each bin satisfies MSR and MTR threshold
2. for each segment, form Meta-SNP via MSPB, and perform EM phasing, compute mhBAF
3. for each segment, aggregate normal bases and tumor bases, compute RDR
4. perform RDR library normalization via RD*(total-normal-bases/total-tumor-bases)

### Output
1. `bb/bin_matrix.alpha.npz`
2. `bb/bin_matrix.beta.npz`
3. `bb/bin_matrix.baf.npz`
3. `bb/bin_matrix.rdr_corr.npz`
4. `bb/bin_matrix.rdr_raw.npz`
5. `bb/bin_matrix.cov.npz`
5. `bb/bin_position.tsv.gz`
6. `phased_snps.tsv.gz`
7. `bb/bulk.bb`

## Clustering
TODO

## Factorization
TODO
