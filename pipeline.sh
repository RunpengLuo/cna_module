#!/bin/bash

set -euo pipefail

# input
SAMPLE=
OUTDIR=
TMPDIR=
LOGDIR=${TMPDIR}/logs
NORMAL_BAM=
TUMOR_BAM=

# preprocessing parameters
q=0
Q=11
d=300
mincov=5
CHROMS=$(seq 1 22)

# binning parameters
MSR=7500
MTR=30
READ_LENGTH=10000
MSPB=20
MBS=1000000

# general parameters
MAXJOBS=4

REGION_BED=/diskmnt/Users2/runpengl/data/chm12v2.0_region.bed
DB_SNP=/diskmnt/Projects/ccRCC_longread/runpengl/vcf/chm13v2.0_dbSNPv155.vcf.gz
REFERENCE=/diskmnt/Projects/ccRCC_longread/runpengl/reference/T2T-CHM13v2.0.fasta

SCRIPT_DIR=/diskmnt/Users2/runpengl/cna_module

mkdir -p ${OUTDIR}
mkdir -p ${TMPDIR}
mkdir -p ${LOGDIR}

echo ${SAMPLE}
echo ${OUTDIR}
########################################
echo "start genotyping on normal sample"
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
    tgt_file="${TMPDIR}/target.${CHROM}.pos.gz"
    bcftools query -f '%CHROM\t%POS\n' -r "${CHROM}" "${DB_SNP}" | gzip -9 > ${tgt_file}
    
    bcftools mpileup "${NORMAL_BAM}" -f "${REFERENCE}" \
    -Ou -a INFO/AD,AD,DP --skip-indels \
    -q ${q} -Q ${Q} -d ${d} -T "${tgt_file}" | \
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
        snp_file="${snp_dir}/${CHROM}.vcf.gz"
        if [[ ! -f ${snp_file} ]]; then
            exit 1
        fi
        bcftools index -f ${snp_file}
        ch_normal_1bed="${TMPDIR}/normal.${CHROM}.1bed"
        bcftools query -f '%CHROM\t%POS\tnormal\t[%AD{0}\t%AD{1}]\n' \
            "${snp_file}" \
            -o "${ch_normal_1bed}"
        cat ${ch_normal_1bed} >> ${normal_1bed}
    done
else
    echo "skip"
fi

echo "allele counting on ${SAMPLE}"
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
        tgt_file="${TMPDIR}/normal.${CHROM}.pos.gz"
        bcftools index -f ${snp_file}
        bcftools query -f '%CHROM\t%POS\n' -r "${CHROM}" "${snp_file}" | gzip -9 > ${tgt_file}

        bcftools mpileup "${TUMOR_BAM}" -f "${REFERENCE}" \
            -Ou -a AD,DP --skip-indels \
            -q ${q} -Q ${Q} -d ${d} -T "${tgt_file}" |
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
        ch_tumor_1bed="${TMPDIR}/${SAMPLE}.${CHROM}.1bed"
        cat ${ch_tumor_1bed} >> ${tumor_1bed}
    done
else
    echo "skip"
fi

########################################
echo "start phasing"
date
phase_dir="${OUTDIR}/phase"
mkdir -p ${phase_dir}
phase_file=${phase_dir}/phased.vcf.gz
if [[ ! -f ${phase_file} ]]; then
    for CHR in $CHROMS; do
        CHROM=chr${CHR}
        ch_phase_file=${TMPDIR}/${CHROM}.phased.vcf.gz
        snp_file="${snp_dir}/${CHROM}.vcf.gz"
        if [[ -f ${ch_phase_file} ]]; then
            echo "${CHROM} exists"
            continue
        fi
        hiphase \
            --bam ${NORMAL_BAM} \
            --reference ${REFERENCE} \
            --vcf ${snp_file} \
            --output-vcf ${ch_phase_file} \
            --threads 2 \
            --ignore-read-groups &>"${LOGDIR}/hiphase.${CHROM}.log" &
        
        while [[ $(jobs -r -p | wc -l) -ge $MAXJOBS ]]; do
            sleep 10
        done
    done

    echo "concat phasing files"
    date

    phase_list_file=${TMPDIR}/phase.list
    >${phase_list_file}

    for CHR in $CHROMS; do
        CHROM=chr${CHR}
        ch_phase_file="${TMPDIR}/${CHROM}.phased.vcf.gz"
        bcftools index -f ${ch_phase_file}
        echo "${ch_phase_file}" >> ${phase_list_file}
    done

    bcftools concat --file-list ${phase_list_file} -Ou \
        | bcftools sort -Oz -o ${phase_file}
    bcftools index -f ${phase_file}
else
    echo "skip"
fi

########################################
echo "run count_reads python script"
date
rdr_dir="${OUTDIR}/rdr"
mkdir -p ${rdr_dir}
test_file=${rdr_dir}/sample_ids.tsv
if [[ ! -f ${test_file} ]]; then
    python -u ${SCRIPT_DIR}/count_reads.py \
        ${SAMPLE} \
        ${REGION_BED} \
        ${baf_dir} \
        ${rdr_dir} \
        ${NORMAL_BAM} \
        ${TUMOR_BAM}
else
    echo "skip"
fi

########################################
echo "run combine_counts python script"
date
bb_dir="${OUTDIR}/bb"
mkdir -p ${bb_dir}
test_file=${bb_dir}/sample_ids.tsv
if [[ ! -f ${test_file} ]]; then
    python -u ${SCRIPT_DIR}/combine_counts.py \
        ${phase_file} \
        ${rdr_dir} \
        ${bb_dir} \
        ${MSR} ${MTR} ${READ_LENGTH} ${MSPB} ${MBS}
else
    echo "skip"
fi

########################################
echo "run cluster_bins python script"
date
bbc_dir="${OUTDIR}/bbc"
# TODO

echo "Done"
date
