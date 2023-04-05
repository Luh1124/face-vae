import os
import argparse
import torch.utils.data as data
import torch.multiprocessing as mp
from logger import Logger
from dataset import FramesDataset, DatasetRepeater, AudioDataset
from distributed import init_seeds, init_dist


def main(proc, args):
    world_size = len(args.gpu_ids)
    init_seeds(not args.benchmark)
    init_dist(proc, world_size)
    if args.data_name == 'vox':
        trainset = DatasetRepeater(FramesDataset(root_dir=args.root_dir), num_repeats=100)
    elif args.data_name == 'lrw':
        trainset = AudioDataset(root_dir=args.root_dir)
    trainsampler = data.distributed.DistributedSampler(trainset)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=trainsampler, drop_last=True)
    # visdom_params = {"server":args.display_server,"port":args.display_port,"env":args.display_env}
    visualizer_params={"kp_size": 5, "draw_border": True, "colormap": "gist_rainbow", "writer_use": False, "writer_name":'running', "use_visdom": False, "visdom_params":{"server":args.display_server,"port":args.display_port,"env":args.display_env}}
    logger = Logger(args.ckp_dir, args.vis_dir, trainloader, args.lr, log_file_name=args.log_file, visualizer_params=visualizer_params)
    if args.ckp > 0:
        logger.load_cpk(args.ckp)
    for i in range(args.num_epochs):
        logger.step()


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    parser = argparse.ArgumentParser(description="face-vid2vid")

    def str2bool(s):
        return s.lower().startswith("t")

    parser.add_argument("--batch_size", default=2, type=int, help="Batch size per GPU")
    parser.add_argument("--benchmark", type=str2bool, default=True, help="Turn on CUDNN benchmarking")
    parser.add_argument("--gpu_ids", default=[0,1], type=eval, help="IDs of GPUs to use")
    parser.add_argument("--lr", default=0.00005, type=float, help="Learning rate")
    parser.add_argument("--num_epochs", default=150, type=int, help="Number of epochs to train")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of data loader threads")
    parser.add_argument("--ckp_dir", type=str, default="ckp_1644_", help="Checkpoint dir")
    parser.add_argument("--vis_dir", type=str, default="vis_1644_", help="Visualization dir")
    parser.add_argument("--ckp", type=int, default=0, help="Checkpoint epoch")
    parser.add_argument("--log_file", type=str, default="log_1644_", help="log file")
    parser.add_argument("--ext", type=str, default="add_debug", help="extension")
    parser.add_argument("--root_dir", type=str, default="/home/luh/lh_8T/datasets/vox1/face-video-preprocessing/vox-png/", help="data_path")
    parser.add_argument("--data_name", type=str, default="vox", help="data_name")

    parser.add_argument("--display_server", type=str, default="130134.46.41", help="data_name")
    parser.add_argument("--display_env", type=str, default="my_voxceleb_", help="data_name")
    parser.add_argument("--display_port", type=int, default=8098, help="data_name")
    parser.add_argument("--use_visdom", type=bool, default=False, help="data_name")

    # parser.add_argument("--display_winsize", type=int, default=256, help="data_name")
    # parser.add_argument("--display_ncols", type=int, default=3, help="data_name")


    args = parser.parse_args()

    args.display_env = args.display_env + args.ext
    args.ckp_dir = args.ckp_dir + args.ext
    args.vis_dir = args.vis_dir + args.ext
    args.log_file = args.log_file + args.ext + '.txt'

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)[1:-1]
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1234"
    mp.spawn(main, nprocs=len(args.gpu_ids), args=(args,))
