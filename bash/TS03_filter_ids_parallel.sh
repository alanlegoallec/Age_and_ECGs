#!/bin/bash
ECG_types=( "features" "resting" "exercising" )
ECG_types=( "features" )
targets=( "Sex" "Age" "Age_group" "Heart_Age" "Heart_Age_SM" "CVD" "CVD_SM" "Diabetic" "Smoker" "SBP" "Cholesterol" "HDL" )
#targets=( "Smoker" )
memory=8G
n_cores=1
time=100
for ECG_type in "${ECG_types[@]}"
do
for target in "${targets[@]}"
do
job_name="t3-$ECG_type-$target.job"
out_file="../eo/t3-$ECG_type-$target.out"
err_file="../eo/t3-$ECG_type-$target.err"
sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -t $time filter_ids.sh $ECG_type $target
done
done
