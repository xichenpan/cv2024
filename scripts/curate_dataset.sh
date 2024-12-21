# Create array of files, filtering out ones already processed
files=()
for file in /fsx/xichenpan/mmc4data/*; do
    # Extract shard name from file path
    shard_name=$(basename "$file" .jsonl)
    # Check if output directory exists
    if [ ! -d "/fsx/xichenpan/mmc4_grouped/$shard_name" ]; then
        files+=("$file")
    fi
done

# Calculate total number of files
total_files=${#files[@]}

# Calculate number of batches (ensuring we don't drop any files)
files_per_batch=1
num_batches=$(( (total_files + files_per_batch - 1) / files_per_batch ))
num_batches_minus_one=$((num_batches - 1))
max_simultaneous_tasks=248

# Create directory structure for batched file lists
mkdir -p /fsx/xichenpan/mmc4_batches
rm -f /fsx/xichenpan/mmc4_batches/*

# Split files into batches
for ((i = 0; i < total_files; i += files_per_batch)); do
    batch_num=$((i / files_per_batch))
    printf "%s\n" "${files[@]:i:files_per_batch}" > "/fsx/xichenpan/mmc4_batches/batch_${batch_num}.txt"
done

# clean up log directory
rm -rf /fsx/xichenpan/mmc4_log/*

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=learnai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH --output=/fsx/xichenpan/mmc4_log/%A_%a.out
#SBATCH --error=/fsx/xichenpan/mmc4_log/%A_%a.err
#SBATCH --array=0-${num_batches_minus_one}%${max_simultaneous_tasks}

source ~/.bashrc
source activate base
conda activate soda

module load cuda/12.1
module load nccl/2.18.3-cuda.12.1
module load nccl_efa/1.24.1-nccl.2.18.3-cuda.12.1

export OMP_NUM_THREADS=12

export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_TOPO_FILE=/opt/aws-ofi-nccl/share/aws-ofi-nccl/xml/p4de-24xl-topo.xml

# Get batch file for this array task
BATCH_FILE="/fsx/xichenpan/mmc4_batches/batch_\${SLURM_ARRAY_TASK_ID}.txt"

echo "Processing batch file: \$BATCH_FILE"
srun python curate_dataset.py --batch_file="\$BATCH_FILE"
EOT

# # Clean up batch files (optional)
# rm -rf /fsx/xichenpan/mmc4_batches
