#!/bin/bash

set -euo pipefail

# input
SAMPLE=
OUTDIR=
TMPDIR=${OUTDIR}/tmp
LOGDIR=${OUTDIR}/log
NORMAL_BAM=
TUMOR_BAM=


SAMPLE_FILE=

# read in a sample_file TSV rather than paths here, first row is normal

# preprocessing parameters
q=0
Q=11
d=300
mincov=5
CHROMS=$(seq 1 22)

# SNP filtering parameters
gamma=0.05
min_ad=1

# phasing parameters
minMAPQ=20
numThreads=8

# binning parameters
MSR=5000
MTR=30
READ_LENGTH=10000
MSPB=20
MBS=1000000 #1e6

# general parameters
MAXJOBS=4

REGION_BED=/diskmnt/Users2/runpengl/data/chm13v2.0_region.bed
DB_SNP=/diskmnt/Projects/ccRCC_longread/runpengl/vcf/chm13v2.0_dbSNPv155.vcf.gz
REFERENCE=/diskmnt/Projects/ccRCC_longread/runpengl/reference/T2T-CHM13v2.0.fasta

SCRIPT_DIR=/diskmnt/Users2/runpengl/cna_module

mkdir -p ${OUTDIR}
mkdir -p ${TMPDIR}
mkdir -p ${LOGDIR}

echo ${SAMPLE}
echo ${OUTDIR}
########################################
echo "genotyping on normal sample"
date
snp_dir="${OUTDIR}/snps"
mkdir -p ${snp_dir}
for CHR in $CHROMS; do
    CHROM=chr${CHR}
    snp_file="${snp_dir}/${CHROM}.vcf.gz"
    if [[ -f ${snp_file} ]]; then
        echo "${CHROM} exists, skip"
        continue
    fi
    bcftools query -f '%CHROM\t%POS\n' -r "${CHROM}" "${DB_SNP}" | gzip -9 > "${TMPDIR}/target.${CHROM}.pos.gz"
    
    bcftools mpileup "${NORMAL_BAM}" -f "${REFERENCE}" \
    -Ou -a INFO/AD,AD,DP --skip-indels \
    -q ${q} -Q ${Q} -d ${d} -T "${TMPDIR}/target.${CHROM}.pos.gz" | \
    bcftools call -m -Ou | \
    bcftools view -v snps -g het -m2 -M2 \
        -i "FMT/DP>=${mincov}" -Oz -o ${snp_file} &>"${LOGDIR}/genotype.${CHROM}.log" &

    while [[ $(jobs -r -p | wc -l) -ge $MAXJOBS ]]; do
        sleep 10
    done
done
wait

baf_dir="${OUTDIR}/baf"
mkdir -p ${baf_dir}

echo "allele counting on normal sample"
date
normal_1bed=${baf_dir}/normal.1bed
if [[ ! -f ${normal_1bed} ]]; then
    >${normal_1bed}
    for CHR in $CHROMS; do
        CHROM=chr${CHR}
        bcftools index -f "${snp_dir}/${CHROM}.vcf.gz"
        bcftools query -f '%CHROM\t%POS\tnormal\t[%AD{0}\t%AD{1}]\n' \
            "${snp_dir}/${CHROM}.vcf.gz" \
            -o "${TMPDIR}/normal.${CHROM}.1bed"
        cat "${TMPDIR}/normal.${CHROM}.1bed" >> ${normal_1bed}
    done
else
    echo "skip"
fi

echo "allele counting on tumor sample=${SAMPLE}"
date
tumor_1bed=${baf_dir}/tumor.1bed
if [[ ! -f ${tumor_1bed} ]]; then
    for CHR in $CHROMS; do
        CHROM=chr${CHR}
        ch_tumor_1bed="${TMPDIR}/${SAMPLE}.${CHROM}.1bed"
        if [[ -f ${ch_tumor_1bed} ]]; then
            echo "${CHROM} exists"
            continue
        fi

        bcftools query -f '%CHROM\t%POS\n' -r "${CHROM}" "${snp_dir}/${CHROM}.vcf.gz" | \
            gzip -9 > "${TMPDIR}/normal.${CHROM}.pos.gz"

        bcftools mpileup "${TUMOR_BAM}" -f "${REFERENCE}" \
            -Ou -a AD,DP --skip-indels \
            -q ${q} -Q ${Q} -d ${d} -T "${TMPDIR}/normal.${CHROM}.pos.gz" | \
            bcftools query \
                -f "%CHROM\t%POS\t${SAMPLE}\t[%AD{0}\t%AD{1}]\n" \
                -o "${ch_tumor_1bed}" &>"${LOGDIR}/count.${SAMPLE}.${CHROM}.log" &

        while [[ $(jobs -r -p | wc -l) -ge $MAXJOBS ]]; do
            sleep 10
        done
    done
    wait

    >${tumor_1bed}
    for CHR in $CHROMS; do
        CHROM=chr${CHR}
        cat "${TMPDIR}/${SAMPLE}.${CHROM}.1bed" >> ${tumor_1bed}
    done
else
    echo "skip"
fi

########################################
echo "form allele-count matrix"
date
allele_dir="${OUTDIR}/allele"
mkdir -p ${allele_dir}
snp_info_file="${allele_dir}/snp_info.tsv.gz"
if [[ ! -f ${snp_info_file} ]]; then
    python -u ${SCRIPT_DIR}/form_snp_matrix.py \
        ${REGION_BED} \
        ${baf_dir} \
        ${allele_dir} \
        "${min_ad}" \
        "${gamma}" \
        ${NORMAL_BAM} ${TUMOR_BAM} &>"${LOGDIR}/form_snp_matrix.log"
else
    echo "skip"
fi

########################################
echo "filter&concat Het SNPs"
date
het_snp_file="${allele_dir}/snps.vcf.gz"
if [[ ! -f ${het_snp_file} ]]; then
    snp_list_file=${TMPDIR}/snps.list
    >${snp_list_file}

    for CHR in $CHROMS; do
        CHROM=chr${CHR}
        echo "${snp_dir}/${CHROM}.vcf.gz" >> ${snp_list_file}
    done
    raw_concat_file=${TMPDIR}/het_snps.raw.vcf.gz
    bcftools concat --output-type z \
        --file-list ${snp_list_file} \
        --output ${raw_concat_file}

    bcftools view -T "${allele_dir}/snps.1pos" \
        -Oz -o ${het_snp_file} ${raw_concat_file}
else
    echo "skip"
fi
        
#######################################
echo "run longphase & HapCUT2"
date
phase_dir="${OUTDIR}/phase"
mkdir -p ${phase_dir}
phase_file=${phase_dir}/phased.vcf.gz
if [[ ! -f ${phase_file} ]]; then
    longphase_linux-x64 phase \
        --bam-file=${NORMAL_BAM} \
        --reference=${REFERENCE} \
        --snp-file=${het_snp_file} \
        --mappingQuality=${minMAPQ} \
        --out-prefix="${TMPDIR}/phased.raw" \
        --pb --threads=${numThreads} &>"${LOGDIR}/longphase.log"
    echo "longphase is finished"
    date

    bash ${SCRIPT_DIR}/filter_unphased.sh ${TMPDIR}/phased.raw.vcf \
        ${phase_dir}/phased.vcf    

    extractHAIRS --bam ${NORMAL_BAM} \
                --VCF ${phase_dir}/phased.vcf \
                --ref ${REFERENCE} \
                --fullprint 1 \
                --realign_variants 0 \
                --out ${TMPDIR}/Normal.fragments.full.txt &>"${LOGDIR}/hairs.normal.log"
    echo "HapCUT2 extractHAIRS on normal sample is finished"
    date

    extractHAIRS --bam ${TUMOR_BAM} \
                --VCF ${phase_dir}/phased.vcf \
                --ref ${REFERENCE} \
                --fullprint 1 \
                --realign_variants 0 \
                --out ${TMPDIR}/Tumor.fragments.full.txt &>"${LOGDIR}/hairs.tumor.log"
    echo "HapCUT2 extractHAIRS on ${SAMPLE} sample is finished"
    date
    
    bgzip "${phase_dir}/phased.vcf"
    python ${SCRIPT_DIR}/hairs.py \
            ${TMPDIR}/Normal.fragments.full.txt \
            ${phase_file} \
            ${phase_dir}/Normal.hairs.tsv.gz

    python ${SCRIPT_DIR}/hairs.py \
            ${TMPDIR}/Tumor.fragments.full.txt \
            ${phase_file} \
            ${phase_dir}/Tumor.hairs.tsv.gz
    date
else
    echo "skip"
fi

# files
# allele/sample_ids.tsv  snp_info.tsv.gz  snp_matrix.dp.npz snp_matrix.alt.npz  snp_matrix.ref.npz  snps.1pos
# phase/phased.vcf.gz Normal.hairs.tsv.gz Tumor.hairs.tsv.gz
# 

########################################
echo "run combine_counts python script"
date
bb_dir="${OUTDIR}/bb"
mkdir -p ${bb_dir}
test_file=${bb_dir}/bulk.bb
if [[ ! -f ${test_file} ]]; then
    python -u ${SCRIPT_DIR}/combine_counts.py \
        ${allele_dir} \
        ${phase_dir} \
        ${bb_dir} \
        ${MSR}
else
    echo "skip"
fi

########################################
echo "run cluster_bins python script"
date
bbc_dir="${OUTDIR}/bbc"
mkdir -p ${bbc_dir}
test_file=${bbc_dir}/bulk.bbc
if [[ ! -f ${test_file} ]]; then
    python -u ${SCRIPT_DIR}/cluster_bins.py \
        ${bb_dir} \
        ${bbc_dir}
else
    echo "skip"
fi

rm -rf ${TMPDIR}
echo "Done"
date
