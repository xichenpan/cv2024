conda install pytorch=2.3.0 pytorch-cuda=12.1 torchvision torchaudio --strict-channel-priority --override-channels -c https://aws-ml-conda.s3.us-west-2.amazonaws.com -c nvidia -c conda-forge -y
pip install -U transformers[deepspeed]
pip install -U accelerate
pip install -U diffusers
pip install datasets==2.21.0
pip install xformers==0.0.26.post1
pip install wandb
pip install timm
pip install OmegaConf
pip install numpy==1.26.4
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.2/flash_attn-2.6.2+cu123torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install pycocotools
pip install -U torchmetrics[image]
pip install lightning
pip install evaluate
pip install scikit-learn
pip install peft
pip install tabulate
pip install piq
pip install sentencepiece
pip install bitsandbytes
pip install qwen-vl-utils
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
rm -rf /data/home/xichenpan/miniforge3/envs/soda/lib/python3.11/site-packages/models