# CUDA_LAUNCH_BLOCKING=1 python train.py --batch_size=2 --gpu_ids=0,1,2,3,4,5,6,7 --ext=mainv5-dl-2-Il-5 --root_dir='/home/momobot/repo/code/2.faceanaimation/dataset/vox1/vox-png'
# python train.py --batch_size=8 --gpu_ids=0,1,2,3 --ext=mainv2-1 --root_dir='/home/lh/repo/datasets/vox-png'  --ckp=1
# python train.py --batch_size=8 --gpu_ids=0,1,2,3 --ext=mainv2-2-kpc --root_dir='/home/lh/repo/datasets/vox-png' 
# python train.py --batch_size=8 --gpu_ids=0,1,2,3 --ext=mainv3-0 --root_dir='/home/lh/repo/datasets/vox-png' 
# python train.py --batch_size=8 --gpu_ids=0,1,2,3 --ext=mainv3-1 --root_dir='/home/lh/repo/datasets/vox-png' 
# python train.py --batch_size=2 --gpu_ids=0,1 --ext=mainv7_vox --data_name='vox' --root_dir='/home/luh/lh_8T/datasets/vox1/face-video-preprocessing/vox-png/' 
# python train.py --batch_size=2 --gpu_ids=0,1 --ext=mainv7_lrw --data_name='lrw' --root_dir='/home/luh/lh_8T/datasets/LRW_Data/LRW_Temp' 
python train.py --batch_size=8 --gpu_ids=0,1,2,3 --ext=mainv8 --data_name='vox' --root_dir='/home/lh/repo/datasets/vox-png' 
