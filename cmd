CUDA_VISIBLE_DEVICES="4,5,6,7" python train_eval_syn.py --cuda --train_dir ../test_images/ --mGPU --restart
python train_eval_syn.py --cuda --train_dir ../eval_images/ --mGPU --eval
