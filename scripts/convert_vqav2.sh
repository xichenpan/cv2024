# Create array of files, filtering out ones already processed
split="val"
files=()
for file in /fsx/xichenpan/vqav2_sharded/$split/*; do
    # Extract shard name from file path
    shard_name=$(basename "$file" .json)
    # Check if output directory exists
    if [ ! -d "/fsx/xichenpan/vqav2_converted/$split/$shard_name" ]; then
        files+=("$file")
    fi
done

max_simultaneous_tasks=128
NUM_FILES_MINUS_ONE=$(( ${#files[@]} - 1 ))

# clean up log directory
rm -rf /fsx/xichenpan/vqav2_log/*

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=learnai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH --output=/fsx/xichenpan/vqav2_log/%A_%a.out
#SBATCH --error=/fsx/xichenpan/vqav2_log/%A_%a.err
#SBATCH --array=0-${NUM_FILES_MINUS_ONE}%${max_simultaneous_tasks}

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

# Get single file for this array task

INPUT_FILE="/fsx/xichenpan/vqav2_sharded/$split/vqav2_shard_\${SLURM_ARRAY_TASK_ID}.json"

echo "Processing file: \$INPUT_FILE"
srun python convert_vqav2.py --file_name="\$INPUT_FILE" --output_dir="/fsx/xichenpan/vqav2_converted/$split"
EOT
