#!/bin/bash
module load ROCm/4.1.0
module load GCC/10.2.0
module load OpenMPI/4.0.5
module load PyTorch/1.8.1
#pip install configobj
#pip install torchsummary
#pip install tensorboardx
#pip install scikit-image==0.16.2
python train_eval_syn.py --cuda --config_file kpn_specs/kpn_config-$1.conf --train_dir /scratch/cl114/yz87/test_images/ --mGPU --restart
