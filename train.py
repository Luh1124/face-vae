import os
import argparse
import torch.utils.data as data
import torch.multiprocessing as mp
from logger import Logger
from dataset import FramesDataset, DatasetRepeater
from distributed import init_seeds, init_dist


def main(proc, args):
    world_size = len(args.gpu_ids)
    init_seeds(not args.benchmark)
    init_dist(proc, world_size)
    trainset = DatasetRepeater(FramesDataset(root_dir=args.root_dir), num_repeats=100)
    trainsampler = data.distributed.DistributedSampler(trainset)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=trainsampler)
    logger = Logger(args.ckp_dir, args.vis_dir, trainloader, args.lr, log_file_name=args.log_file)
    if args.ckp > 0:
        logger.load_cpk(args.ckp)
    for i in range(args.num_epochs):
        logger.step()


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    parser = argparse.ArgumentParser(description="face-vid2vid")

    def str2bool(s):
        return s.lower().startswith("t")

    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--benchmark", type=str2bool, default=True, help="Turn on CUDNN benchmarking")
    parser.add_argument("--gpu_ids", default=[0,1,2], type=eval, help="IDs of GPUs to use")
    parser.add_argument("--lr", default=0.00005, type=float, help="Learning rate")
    parser.add_argument("--num_epochs", default=150, type=int, help="Number of epochs to train")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of data loader threads")
    parser.add_argument("--ckp_dir", type=str, default="ckp_1644_", help="Checkpoint dir")
    parser.add_argument("--vis_dir", type=str, default="vis_1644_", help="Visualization dir")
    parser.add_argument("--ckp", type=int, default=0, help="Checkpoint epoch")
    parser.add_argument("--log_file", type=str, default="log_1644_.txt", help="log file")
    parser.add_argument("--ext", type=str, default="add", help="extension")
    parser.add_argument("--root_dir", type=str, default="/home/lh/repo/datasets/face-video-preprocessing/vox-png", help="data_path")


    args = parser.parse_args()

    args.ckp_dir = args.ckp_dir + args.ext
    args.vis_dir = args.vis_dir + args.ext
    args.log_file = os.path.split(args.log_file)[0] + args.ext + '.txt'

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)[1:-1]
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    mp.spawn(main, nprocs=len(args.gpu_ids), args=(args,))
