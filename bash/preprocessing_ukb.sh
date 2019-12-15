#!/bin/bash
#/n/groups/patel/uk_biobank/pheno_29483ukbfetch -bukb29483.bulk -s$s -m$m -of1
s=0
while [ $s -le 100 ]
do
m=$((s+10))
/n/groups/patel/uk_biobank/pheno_29483/ukbfetch -bukb29483.bulk -s$s -m$m -of1
((s+=10))
echo $s
echo $m
done
