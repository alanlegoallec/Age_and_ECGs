#!/bin/bash
targets=( "Sex" "Age" "Age_group" "Heart_Age" "Heart_Age_SM" "CVD" "CVD_SM" "Diabetic" "Smoker" "SBP" "Cholesterol" "HDL" )
algorithms=( "ElasticNet" "KNN" "Bayesian" "SVM" "RandomForest" "GBM" "XGB" "NeuralNetwork" )
n_seeds_start=420
n_seeds_end=500
memory=8G
n_cores=1
for seed in $(seq $n_seeds_start $n_seeds_end)
do
for target in "${targets[@]}"
do
for algorithm in "${algorithms[@]}"
do
time=5
if [ $algorithm = "SVM" ]; then
  time=$(( 2*$time ))
fi
if [ $algorithm = "SVM" ] || [ $algorithm = "GBM" ] && [ $target = "Smoker" ]; then
  time=$(( 2*$time ))
fi
job_name="t4f-$target-$algorithm-$seed.job"
out_file="../eo/t4f-$target-$algorithm-$seed.out"
err_file="../eo/t4f-$target-$algorithm-$seed.err"
sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -t $time tuning_f.sh $target $algorithm $seed
done
done
done

