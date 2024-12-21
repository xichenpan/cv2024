## Env Setup
```bash
conda create -n instructdiff python=3.10
conda activate instructdiff
bash setup.sh
```

## Data Construction
### Instruction Tuning Data
First download the mmc4data from [here](https://github.com/allenai/mmc4/blob/main/scripts/fewer_faces_corev3.sh)

```bash
git clone https://github.com/allenai/mmc4.git
cd mmc4
bash scripts/fewer_faces_corev3.sh /path/to/output
cd /path/to/output
unzip "*.zip"
rm -rf *.zip
```

Then modify the `scripts/curate_dataset.sh` to match the path to your mmc4data.
```bash
bash scripts/curate_dataset.sh
```

### VQAV2 Data
download the vqav2 data
```bash
mkdir vqav2
cd vqav2
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip
unzip "*.zip"
rm -rf *.zip
```

Then split the vqav2 data into train and val
```bash
python scripts/split_vqav2.py --data_dir /path/to/vqav2 --output_dir /path/to/vqav2_split --shard_size 1000
```

Then modify the `scripts/convert_vqav2.sh` to match the path to your vqav2 data.
```bash
bash scripts/convert_vqav2.sh
```

## Train
### Alignment
```bash
OMP_NUM_THREADS=12 torchrun --nproc-per-node=8 train.py --run_name alignment --config_file siglip_384_luminanexticl_1024_flow_proj.yaml
```

### End-to-end
```bash
OMP_NUM_THREADS=12 torchrun --nproc-per-node=8 train.py --run_name end2end --config_file siglip_384_luminanexticl_1024_flow_inst.yaml
```

### SFT
```bash
OMP_NUM_THREADS=12 torchrun --nproc-per-node=8 train.py --run_name sft --config_file siglip_384_luminanexticl_1024_flow_proj.yaml
```

## Evaluation
```bash
python app.py --resume_from_checkpoint /path/to/checkpoint
```