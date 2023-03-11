import argparse
from models import EFE_6 as EFE
from models import AFE, CKD, HPE_EDE, MFE, Generator
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import imageio
import os
from skimage import io, img_as_float32, img_as_ubyte
from utils import transform_kp, transform_kp_with_new_pose
from logger import Visualizer
import torchvision
import math
from utils import apply_imagenet_normalization
from tqdm import tqdm, trange
import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import pandas as pd

class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)
        self.idx_tensor = torch.FloatTensor(list(range(num_bins))).unsqueeze(0).cuda()
        self.n_bins = num_bins
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        real_yaw = self.fc_yaw(x)
        real_pitch = self.fc_pitch(x)
        real_roll = self.fc_roll(x)
        real_yaw = torch.softmax(real_yaw, dim=1)
        real_pitch = torch.softmax(real_pitch, dim=1)
        real_roll = torch.softmax(real_roll, dim=1)
        real_yaw = (real_yaw * self.idx_tensor).sum(dim=1)
        real_pitch = (real_pitch * self.idx_tensor).sum(dim=1)
        real_roll = (real_roll * self.idx_tensor).sum(dim=1)
        real_yaw = (real_yaw - self.n_bins // 2) * 3 * np.pi / 180
        real_pitch = (real_pitch - self.n_bins // 2) * 3 * np.pi / 180
        real_roll = (real_roll - self.n_bins // 2) * 3 * np.pi / 180

        return real_yaw, real_pitch, real_roll

@torch.no_grad()
def eval(args):
    g_models = {"afe": AFE(), "ckd": CKD(), "hpe_ede": HPE_EDE(), "mfe": MFE(), "generator": Generator()}
    ckp_path = os.path.join(args.ckp_dir, "%s-checkpoint.pth.tar" % str(args.ckp).zfill(8))
    checkpoint = torch.load(ckp_path, map_location=torch.device("cpu"))
    for k, v in g_models.items():
        v.cuda()
        v.load_state_dict(checkpoint[k])
        v.eval()
    output_frames = []
    if args.source == "r":
        frames = sorted(os.listdir(args.driving))[: args.num_frames]
        num_frames = len(frames)
        video_array = [img_as_float32(io.imread(os.path.join(args.driving, frames[idx]))) for idx in range(num_frames)]
        s = np.array(video_array[0], dtype="float32").transpose((2, 0, 1))
        s = torch.from_numpy(s).cuda().unsqueeze(0)
        fs = g_models["afe"](s)
        kp_c = g_models["ckd"](s)
        yaw_s, pitch_s, roll_s, t_s, delta_s = g_models["hpe_ede"](s)
        kp_s, Rs = transform_kp(kp_c, yaw_s, pitch_s, roll_s, t_s, delta_s)
        for img in video_array[1:]:
            img = np.array(img, dtype="float32").transpose((2, 0, 1))
            img = torch.from_numpy(img).cuda().unsqueeze(0)
            yaw, pitch, roll, t, delta = g_models["hpe_ede"](img)
            kp_d, Rd = transform_kp(kp_c, yaw, pitch, roll, t, delta)
            deformation, occlusion = g_models["mfe"](fs, kp_s, kp_d, Rs, Rd)
            generated_d = g_models["generator"](fs, deformation, occlusion)
            generated_d = torch.cat((img, generated_d), dim=3)
            # generated_d = F.interpolate(generated_d, scale_factor=0.5)
            generated_d = generated_d.squeeze(0).data.cpu().numpy()
            generated_d = np.transpose(generated_d, [1, 2, 0])
            generated_d = generated_d.clip(0, 1)
            generated_d = (255 * generated_d).astype(np.uint8)
            output_frames.append(generated_d)
    elif args.source == "f":
        frames = sorted(os.listdir(args.driving))[: args.num_frames]
        num_frames = len(frames)
        video_array = [img_as_float32(io.imread(os.path.join(args.driving, frames[idx]))) for idx in range(num_frames)]
        for img in video_array:
            img = np.array(img, dtype="float32").transpose((2, 0, 1))
            img = torch.from_numpy(img).cuda().unsqueeze(0)
            fs = g_models["afe"](img)
            kp_c = g_models["ckd"](img)
            yaw, pitch, roll, t, delta = g_models["hpe_ede"](img)
            kp_s, Rs = transform_kp(kp_c, yaw, pitch, roll, t, delta)
            kp_d, Rd = transform_kp_with_new_pose(kp_c, yaw, pitch, roll, t, delta, 0 * yaw, 0 * pitch, 0 * roll)
            deformation, occlusion = g_models["mfe"](fs, kp_s, kp_d, Rs, Rd)
            generated_d = g_models["generator"](fs, deformation, occlusion)
            generated_d = torch.cat((img, generated_d), dim=3)
            # generated_d = F.interpolate(generated_d, scale_factor=0.5)
            generated_d = generated_d.squeeze(0).data.cpu().numpy()
            generated_d = np.transpose(generated_d, [1, 2, 0])
            generated_d = generated_d.clip(0, 1)
            generated_d = (255 * generated_d).astype(np.uint8)
            output_frames.append(generated_d)
    else:
        s = img_as_float32(io.imread(args.source))[:, :, :3]
        s = np.array(s, dtype="float32").transpose((2, 0, 1))
        s = torch.from_numpy(s).cuda().unsqueeze(0)
        s = F.interpolate(s, size=(256, 256))
        fs = g_models["afe"](s)
        kp_c = g_models["ckd"](s)
        yaw, pitch, roll, t, delta = g_models["hpe_ede"](s)
        kp_s, Rs = transform_kp(kp_c, yaw, pitch, roll, t, delta)
        frames = sorted(os.listdir(args.driving))[: args.num_frames]
        num_frames = len(frames)
        video_array = [img_as_float32(io.imread(os.path.join(args.driving, frames[idx]))) for idx in range(num_frames)]
        for img in video_array:
            img = np.array(img, dtype="float32").transpose((2, 0, 1))
            img = torch.from_numpy(img).cuda().unsqueeze(0)
            yaw, pitch, roll, t, delta = g_models["hpe_ede"](img)
            kp_d, Rd = transform_kp(kp_c, yaw, pitch, roll, t, delta)
            deformation, occlusion = g_models["mfe"](fs, kp_s, kp_d, Rs, Rd)
            generated_d = g_models["generator"](fs, deformation, occlusion)
            generated_d = torch.cat((img, generated_d), dim=3)
            generated_d = generated_d.squeeze(0).data.cpu().numpy()
            generated_d = np.transpose(generated_d, [1, 2, 0])
            generated_d = generated_d.clip(0, 1)
            generated_d = (255 * generated_d).astype(np.uint8)
            output_frames.append(generated_d)
    imageio.mimsave(args.output, output_frames)

import re
imageio.plugins.freeimage.download()
def demo(args):
    hp = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66).cuda()
    hp.load_state_dict(torch.load("hopenet_robust_alpha1.pkl", map_location=torch.device("cpu")))
    for parameter in hp.parameters():
        parameter.requires_grad = False
    g_models = {"efe": EFE(), "afe": AFE(), "ckd": CKD(), "hpe_ede": HPE_EDE(), "mfe": MFE(), "generator": Generator()}
    ckp_path = os.path.join(args.ckp_dir, "%s-checkpoint.pth.tar" % str(args.ckp).zfill(8))
    checkpoint = torch.load(ckp_path, map_location=torch.device("cpu"))
    for k, v in g_models.items():
        v.cuda()
        v.load_state_dict(checkpoint[k])
        v.eval()

    video_paths = sorted([os.path.join(args.test_path, p) for p in os.listdir(args.test_path)])

    loss_list = {'file_name': [], 'frame_number': [], "L1": [], "MSE": [], "PSNR": [], "SSIM": []}

    for video_idx, video_path in tqdm(enumerate(video_paths), total=len(video_paths)):
        # if video_idx > 2:
        #     break
        video_name = os.path.basename(video_path)
        predictions = []
        
        img_names = sorted(os.listdir(video_path))
        num_frames = len(img_names)
        video_array = [img_as_float32(io.imread(os.path.join(video_path, img_names[idx])))[:, :, :3] for idx in range(num_frames)]
        sn = img_names[0]
        dns = img_names[:]
        
        s = img_as_float32(io.imread(os.path.join(video_path, sn)))
        s = np.array(s, dtype="float32").transpose((2, 0, 1))
        s = torch.from_numpy(s).cuda().unsqueeze(0)
        s = F.interpolate(s, size=(256, 256))


        with torch.no_grad():
            fs = g_models["afe"](s)
            kp_c = g_models["ckd"](s)
            _, _, _, t, scale = g_models["hpe_ede"](s)
            hp.eval()
            yaw, pitch, roll = hp(F.interpolate(apply_imagenet_normalization(s), size=(224, 224)))
            # kp_s, Rs = transform_kp(kp_c, yaw, pitch, roll, t, scale)
            # kp_s, _, _, _, _ = g_models["efe"](s, None, kp_s)
            delta_s, _, _, _, _ = g_models["efe"](s, None, kp_c)
            kp_s, Rs = transform_kp(kp_c+delta_s, yaw, pitch, roll, t, scale)
        
        s_np = s.data.cpu()
        s_np = np.transpose(s_np, [0, 2, 3, 1])

        for frame_idx, drin in enumerate(dns):
            dri = img_as_float32(io.imread(os.path.join(video_path, drin)))
            save_img_name = os.path.join(args.png_dir, video_name + f'{frame_idx}.png')
            if os.path.exists(save_img_name):
                generated_d_np = img_as_float32(io.imread(save_img_name))
            else:
                img = np.array(dri, dtype="float32").transpose((2, 0, 1))
                img = torch.from_numpy(img).cuda().unsqueeze(0)
                _, _, _, t, scale = g_models["hpe_ede"](img)
                with torch.no_grad():
                    hp.eval()
                    yaw, pitch, roll = hp(F.interpolate(apply_imagenet_normalization(img), size=(224, 224)))
                    # kp_c_d = g_models["ckd"](img)

                    # delta = delta
                    delta_d, _, _, _, _ = g_models["efe"](img, None, kp_c)
                kp_d, Rd = transform_kp(kp_c+delta_d, yaw, pitch, roll, t, scale)
                deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d, Rs, Rd)
                generated_d = g_models["generator"](fs, deformation, occlusion)
                
                generated_d_np = generated_d.data.cpu().numpy().transpose([0, 2, 3, 1])[0]

                pre_image = img_as_ubyte(generated_d_np)
                predictions.append(pre_image)

                cv2.imwrite(save_img_name, pre_image[:, :, ::-1])

            loss_list['file_name'].append(video_name)
            loss_list['frame_number'].append(frame_idx)
            loss_list['L1'].append(np.abs(generated_d_np - dri))
            loss_list['MSE'].append(mean_squared_error(generated_d_np, dri))
            loss_list['PSNR'].append(peak_signal_noise_ratio(generated_d_np, dri))
            loss_list['SSIM'].append(structural_similarity(generated_d_np, dri, channel_axis=2))

        if not os.path.exists(os.path.join(args.video_dir, video_name)):
            imageio.mimsave(os.path.join(args.video_dir, video_name),  predictions, fps=25)

    print("The length of pngs:", len(loss_list['L1']))
    l1loss_str = f"L1 loss: {np.mean(loss_list['L1'])} \n \
        MSE loss: {(np.mean(loss_list['MSE']))} \n \
        PSNR loss: {(np.mean(loss_list['PSNR']))} \n \
        SSIM loss: {(np.mean(loss_list['SSIM']))}"
    print(l1loss_str)

    return pd.DataFrame(loss_list)


def cal_metrics(args):
    if os.path.exists(args.out_file):
        df = pd.read_pickle(args.out_file)
    else: 
        # df = demo(args)
        df = read_and_cal(args)
        df.to_pickle(args.out_file) 
    
    df = df.sort_values(by=['file_name', 'frame_number'])
    L1, MSE, PSNR, SSIM = [],[],[],[]

    col_mean = df[["L1","MSE","PSNR","SSIM"]].mean()
    col_mean["Name"]="Summary"
    df = df.append(col_mean, ignore_index=True)
    print(col_mean)
    # df.to_pickle(args.out_file) 

def read_and_cal(args):
    loss_list = {'file_name': [], 'frame_number': [], "L1": [], "MSE": [], "PSNR": [], "SSIM": []}
    video_paths = sorted([os.path.join(args.test_path, p) for p in os.listdir(args.test_path)])
    for video_idx, video_path in tqdm(enumerate(video_paths), total=len(video_paths)):
        video_name = os.path.basename(video_path)
        img_names = sorted(os.listdir(video_path))
        num_frames = len(img_names)
        # video_array = [img_as_float32(io.imread(os.path.join(video_path, img_names[idx])))[:, :, :3] for idx in range(num_frames)]
        for frame_idx, drin in enumerate(img_names):
            dri = img_as_float32(io.imread(os.path.join(video_path, drin)))
            save_img_name = os.path.join(args.png_dir, video_name + f'{frame_idx}.png')
            if os.path.exists(save_img_name):
                generated_d_np = img_as_float32(io.imread(save_img_name))
            
            loss_list['file_name'].append(video_name)
            loss_list['frame_number'].append(frame_idx)
            loss_list['L1'].append(np.abs(generated_d_np - dri))
            loss_list['MSE'].append(mean_squared_error(generated_d_np, dri))
            loss_list['PSNR'].append(peak_signal_noise_ratio(generated_d_np, dri))
            loss_list['SSIM'].append(structural_similarity(generated_d_np, dri, channel_axis=2))
    
    print("The length of pngs:", len(loss_list['L1']))
    # l1loss_str = f"L1 loss: {np.mean(loss_list['L1'])} \n \
    #     MSE loss: {(np.mean(loss_list['MSE']))} \n \
    #     PSNR loss: {(np.mean(loss_list['PSNR']))} \n \
    #     SSIM loss: {(np.mean(loss_list['SSIM']))}"
    # print(l1loss_str)

    return pd.DataFrame(loss_list)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    parser = argparse.ArgumentParser(description="face-vid2vid")

    def str2bool(s):
        return s.lower().startswith("t")

    parser.add_argument("--ckp_dir", type=str, default="ckp_1644_mainv9-dl5-lkpc-notanh", help="Checkpoint dir")
    parser.add_argument("--ckp", type=int, default=77, help="Checkpoint epoch")
    parser.add_argument("--test_path", type=str, default="/home/lh/repo/datasets/vox-png/test", help="test_path")
    parser.add_argument("--log_dir", type=str, default='/home/lh/repo3/lh/repo/lh_code/metric_nerfface', help="log_path")
    parser.add_argument("--method", type=str, default='facevae_mainv9-dl5-lkpc-notanh-77', help="log_path")


    args = parser.parse_args()
    # eval(args)

    os.makedirs(args.log_dir, exist_ok=True)
    args.video_dir = os.path.join(args.log_dir, args.method, 'video')
    args.png_dir = os.path.join(args.log_dir, args.method, 'png')
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.png_dir, exist_ok=True)
    args.out_file = os.path.join(args.log_dir, args.method, "loss_list.pkl")
    
    cal_metrics(args)
