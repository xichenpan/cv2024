# Create array of files, filtering out ones already processed
shard_size=100
python split_coco.py --shard_size=$shard_size

files=()
for file in /fsx/xichenpan/coco/val_shard_*; do
    # Extract shard name from file path
    shard_name=$(basename "$file")
    # Check if output directory exists
    if [ ! -d "/fsx/xichenpan/eval/coco/$shard_name" ]; then
        files+=("$file")
    fi
done

max_simultaneous_tasks=128
NUM_FILES_MINUS_ONE=$(( ${#files[@]} - 1 ))

# clean up log directory
rm -rf /fsx/xichenpan/coco_log/*

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=learnai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH --output=/fsx/xichenpan/coco_log/%A_%a.out
#SBATCH --error=/fsx/xichenpan/coco_log/%A_%a.err
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
INPUT_FILE="/fsx/xichenpan/coco/val_shard_\${SLURM_ARRAY_TASK_ID}"

echo "Processing file: \$INPUT_FILE"
srun python sample_coco_lumina.py --file_name="\$INPUT_FILE"
EOT
