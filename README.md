## Binning with bed file
count reads + combine counts

### Input
1. `baf` directory, 1-indexed snp positions
2. 0-indexed bed region file, centromeres and seg-dup regions are excluded
3. normal bam, tumor bam, need for rdr
4. 

### count-reads Output
1. dataframe
    1. #CHR, POS, START, END
2. numpy matrix SNP by SAMPLE (starts with normal)
    1. ref matrix
    2. alt matrix
    3. read depth
3. total-reads count tsv file
    1. SAMPLE, total read counts

### combine-counts Output
1. 1-based bb file

# cna_module


## Final output
1. bulk.bbc
2. bulk.seg
