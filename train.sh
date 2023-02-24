# CUDA_LAUNCH_BLOCKING=1 python train.py --batch_size=3 --gpu_ids=0,1,2,3,4,5,6,7 --ext=add --root_dir='/home/momobot/repo/code/2.faceanaimation/dataset/vox1/vox-png' --ckp=44
# python train.py --batch_size=8 --gpu_ids=0,1,2,3 --ext=mainv2-1 --root_dir='/home/lh/repo/datasets/vox-png'  --ckp=1
# python train.py --batch_size=8 --gpu_ids=0,1,2,3 --ext=mainv2-2-kpc --root_dir='/home/lh/repo/datasets/vox-png' 
# python train.py --batch_size=8 --gpu_ids=0,1,2,3 --ext=mainv3-0 --root_dir='/home/lh/repo/datasets/vox-png' 
python train.py --batch_size=8 --gpu_ids=0,1,2,3 --ext=mainv3-1-kp --root_dir='/home/lh/repo/datasets/vox-png' --ckp=0
