#!/bin/bash

# Adaptive MTL Training Script for Cityscapes
# This script runs trainer_adaptive.py with FairGrad method

# Configuration
method=fairgrad
alpha=2.0        # For FairGrad
gamma=1e-5       # For FAMO imbalance detector
beta=0.85        # For Vargrad
weights_threshold=1.5  # Threshold for imbalance detection
batch_size=8
n_epochs=200
lr=1e-4
seed=1

# Create logs directory if it doesn't exist
mkdir -p trainlogs

# Function to run training with given parameters
run_training() {
    local threshold=$1
    local use_thresh=$2
    local seed=$3
    
    timestamp=$(date +"%Y%m%d_%H%M%S")
    
    # Generate log filename
    if [ "$use_thresh" = "true" ]; then
        thresh_suffix="thresh"
    else
        thresh_suffix="always"
    fi
    
    log_file="trainlogs/fairgrad_alpha${alpha}_${thresh_suffix}${threshold}_sd${seed}_bs${batch_size}_${timestamp}.log"
    cmd="python -u trainer_vargrad.py --method=$method --alpha=$alpha --beta=$beta --weights_threshold=$threshold --use_threshold=$use_thresh --seed=$seed --batch-size=$batch_size --n-epochs=$n_epochs --lr=$lr --N_steps=1 --data-path=./dataset"
    
    echo "Running: $cmd"
    echo "Log file: $log_file"
    echo "----------------------------------------"
    
    nohup $cmd > $log_file 2>&1 &
    
    # Wait a bit before starting next job
    sleep 5
}

# Main execution
echo "Starting Adaptive FairGrad Training Experiments"
echo "=============================================="

# Experiment: FairGrad with always MTL (no threshold)
echo "Experiment: FairGrad with always MTL"
run_training $weights_threshold "false" $seed

echo "All experiments started!"
echo "Check trainlogs/ directory for log files"
echo "Use 'ps aux | grep trainer_adaptive' to see running processes"
echo "Use 'tail -f trainlogs/*.log' to monitor logs" 
