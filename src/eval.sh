#!/bin/bash

thresholds=(-10000 -1000 -900 -800 -700 -600 -500 -400 -300 -200 -100 0 100 200 300 400 500)
num_gaussians=(1 2 3 5 8 9 16 20)
train_iterations=(16 32 64 128 200)

run_python() {
    threshold=$1
    # env variable for python3.9
    output=$(/usr/bin/python3 voice_GMM.py --hard-decision-threshold="$threshold" | tail -n 1)

    echo "$output"
}

for threshold in "${thresholds[@]}"; do
    echo "Running with threshold: $threshold"
    output=$(run_python "$threshold")
    tp=$(echo "$output" | grep -oE 'tp:[[:digit:]]+')
    tn=$(echo "$output" | grep -oE 'tn:[[:digit:]]+')
    fp=$(echo "$output" | grep -oE 'fp:[[:digit:]]+')
    fn=$(echo "$output" | grep -oE 'fn:[[:digit:]]+')
    accuracy=$(echo "$output" | grep -oE 'accuracy:[[:digit:]]+\.[[:digit:]]+')
    echo "TP: $tp, TN: $tn, FP: $fp, FN: $fn, Accuracy: $accuracy"
    echo "$threshold,$tp,$tn,$fp,$fn,$accuracy" >> results.csv
done
