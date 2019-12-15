#!/bin/bash
targets=( "Sex" "Age" "Age_group" "Heart_Age" "Heart_Age_SM" "CVD" "CVD_SM" "Diabetic" "Smoker" "SBP" "Cholesterol" "HDL" )
ECG_types=( "resting" "exercising" )
ECG_types=( "resting" )
algorithms=( "Conv1D" "SimpleRNN" "LSTM" "GRU" )
n_seeds_start=3
n_seeds_end=150
memory=8G
n_cores=1
time=30
for seed in $(seq $n_seeds_start $n_seeds_end)
do
for target in "${targets[@]}"
do
for ECG_type in "${ECG_types[@]}"
do
for algorithm in "${algorithms[@]}"
do
job_name="t4-$target-$ECG_type-$algorithm-$seed.job"
out_file="../eo/t4-$target-$ECG_type-$algorithm-$seed.out"
err_file="../eo/t4-$target-$ECG_type-$algorithm-$seed.err"
sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -t $time tuning.sh $target $ECG_type $algorithm $seed
done
done
done
done
