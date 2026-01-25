#!/bin/bash
####### Reserve computing resources #############
#SBATCH --time=3:00:00 
#SBATCH --job-name=test_combined_hyps_v2 
#SBATCH --account=st-li1210-1
#SBATCH --nodes=1        
#SBATCH --ntasks=1   
#SBATCH --cpus-per-task=3                         
#SBATCH --mem=15G
#SBATCH --output=/scratch/st-li1210-1/pearl/karyotype-detector/logs/yolov5/test_combined_hyps_v2_output.txt
#SBATCH --error=/scratch/st-li1210-1/pearl/karyotype-detector/logs/yolov5/test_combined_hyps_v2_error.txt
#SBATCH --mail-user=jinjpark@student.ubc.ca
#SBATCH --mail-type=ALL

module load gcc

# Activate conda environment
source ~/.bashrc
conda activate yolov5

# Set working directory
export main=/scratch/st-li1210-1/pearl/karyotype-detector
cd $main/models/yolov5
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# List of datasets to test
DATASETS=(
    "data/test_norm_data.yaml"
)

# Test parameters
# resizes to that size
IMGSZ=1024

mkdir -p runs/test

# Loop over each dataset
for DATA_YAML in "${DATASETS[@]}"; do
    echo "=== Running test for $DATA_YAML ==="
    python val_modified.py \
        --weights runs/train/final_combined_hyps_v2/weights/best.pt \
        --data $DATA_YAML \
        --task test \
        --imgsz $IMGSZ \
        --save-txt \
        --save-conf \
        --project runs/test \
        --name combined_hyps_v2_on_chrom2
done

# Deactivate environment
conda deactivate

