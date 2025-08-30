#!/bin/bash

# input and output are uncompressed
INVCF=$1
OUTVCF=$2

total=0
kept=0

awk '
BEGIN {OFS="\t"}
/^#/ {print > "'"$OUTVCF"'"; next}

{
total++
split($9, fmt, ":")
ps_idx = 0
for (i = 1; i <= length(fmt); i++) {
    if (fmt[i] == "PS") {
    ps_idx = i
    break
    }
}
if (ps_idx == 0) next  # skip if no PS in FORMAT

keep = 1
for (j = 10; j <= NF; j++) {
    split($j, sample, ":")
    if (sample[ps_idx] == "." || sample[ps_idx] == "") {
    keep = 0
    break
    }
}

if (keep) {
    kept++
    print > "'"$OUTVCF"'"
}
}
END {
print "Total SNPs before filtering: " total > "/dev/stderr"
print "Total SNPs after filtering:  " kept > "/dev/stderr"
print "Removed:                      " total - kept > "/dev/stderr"
}
' "$INVCF"
