# rm -r /scratch/yz87/models/
# rm -r /scratch/yz87/test_images/
# rm -r /scratch/yz87/eval_images/
#rm DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip 
#rm disclaimer.txt
#rm -r __MACOSX
#rm -r original_high_fps_videos
mkdir /home/zy42/Single_Photon/dataset/test_images/
mkdir /home/zy42/Single_Photon/dataset/eval_images/
cd /home/zy42/Single_Photon/dataset
if [ -f "DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip"  ]; then
    echo "Dataset downloaded"
else
    wget http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip
    unzip DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip
fi
cd /home/zy42/Single_Photon/kernel-prediction-networks-PyTorch/
python dataset_test.py
# rm -r /home/zy42/Single_Photon/dataset/test_images/.DS_Store/
# mkdir /home/zy42/Single_Photon/dataset/models
# cd /home/zy42/Single_Photon/kernel-prediction-networks-PyTorch/
# python train_eval_syn.py --cuda --config_file kpn_128/kpn_config-${job}.conf --train_dir /scratch/yz87/test_images/ --mGPU --restart
# wait
# #srun --exclusive --nodes 1 --ntasks 1 python train_eval_syn.py --cuda --config_file kpn_specs/kpn_config-6.conf --train_dir /scratch/yz87/test_images/ --mGPU --restart 
# #srun --exclusive --nodes 1 --ntasks 1 python train_eval_syn.py --cuda --config_file kpn_specs/kpn_config-3.conf --train_dir /scratch/yz87/test_images/ --mGPU  
