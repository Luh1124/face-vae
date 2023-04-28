# 读取图片列表，批量生成

import argparse
from models import EFE_6 as EFE
from models import AFE, CKD, HPE_EDE, MFE
from models import Generator_FPN as Generator
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
import sys
import glob
import cv2
from tqdm import tqdm, trange
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from imageio import mimread
from skimage.color import gray2rgb

def read_video(name):
    """
    Read video which can be:
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array([img_as_float32(io.imread(os.path.join(name, str(frames[idx], encoding="utf-8")))) for idx in range(num_frames)])
    elif name.lower().endswith(".gif") or name.lower().endswith(".mp4"):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array

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
    

def init_model(args):
    hp = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66).cuda()
    hp.load_state_dict(torch.load(args.hp_pickle_path, map_location=torch.device("cpu")))
    hp.eval()
    for parameter in hp.parameters():
        parameter.requires_grad = False
    g_models = {"efe": EFE(), "afe": AFE(), "ckd": CKD(), "hpe_ede": HPE_EDE(), "mfe": MFE(), "generator": Generator()}
    ckp_path = os.path.join(args.ckp_dir, "%s-checkpoint.pth.tar" % str(args.ckp).zfill(8))
    checkpoint = torch.load(ckp_path, map_location=torch.device("cpu"))
    for k, v in g_models.items():
        v.cuda()
        v.load_state_dict(checkpoint[k])
        v.eval()

    return hp, g_models

@torch.no_grad()
def reconstruction(hp, g_models, img_dir, img_size, save_dir):
    MSE = []
    PSNR = []
    SSIM = []
    L1 = []
    
    video_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    for ind, video_path in enumerate(video_paths):
        video_save_dir = os.path.join(save_dir, os.path.basename(video_path))
        img_paths = sorted(glob.glob(os.path.join(video_path, "*.png")))
        s = img_as_float32(io.imread(img_paths[0]))[:, :, :3]
        s = np.array(s, dtype="float32").transpose((2, 0, 1))
        s = torch.from_numpy(s).unsqueeze(0).cuda()
        s = F.interpolate(s, size=(img_size, img_size), mode="bilinear", align_corners=False)
        fs = g_models["afe"](s)
        kp_c = g_models["ckd"](s)
        _, _, _, t_s, scale_s = g_models["hpe_ede"](s)
        yaw_s, pitch_s, roll_s = hp(F.interpolate(apply_imagenet_normalization(s), size=(224, 224)))
        delta_s, _, _, _, _ = g_models["efe"](s, None, kp_c)
        kp_s, Rs = transform_kp(kp_c+delta_s, yaw_s, pitch_s, roll_s, t_s, scale_s)
        output_frames = []
        for idx, dri in tqdm(enumerate(img_paths), total=len(img_paths), desc=f"{ind}/{len(video_paths)}:"):
            dri_image = io.imread(dri)[:, :, :3]
            # dri_image = cv2.cvtColor(cv2.imread(dri)[:, :, :3], cv2.COLOR_BGR2RGB)
            img = img_as_float32(dri_image)
            img = np.array(img, dtype="float32").transpose((2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            img = F.interpolate(img, size=(img_size, img_size), mode="bilinear", align_corners=False)
            _, _, _, t_d, scale_d = g_models["hpe_ede"](img)
            # kp_c_d = g_models["ckd"](img)
            yaw_d, pitch_d, roll_d = hp(F.interpolate(apply_imagenet_normalization(img), size=(224, 224)))
            delta_d, _, _, _, _ = g_models["efe"](img, None, kp_c)
            kp_d, Rd = transform_kp(kp_c+delta_d, yaw_d, pitch_d, roll_d, t_d, scale_d)
            deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d, Rs, Rd)
            generated_d, _, _ = g_models["generator"](fs, deformation, occlusion)
            
            L1.append(torch.mean(torch.abs(img - generated_d)).item())
            
            generated_d = generated_d.squeeze(0).data.cpu().numpy()
            generated_d = np.transpose(generated_d, [1, 2, 0])
            generated_d = generated_d.clip(0, 1)
            pre_image = (255 * generated_d).astype(np.uint8)
            output_frames.append(pre_image)

            MSE.append(mean_squared_error(pre_image, dri_image))
            PSNR.append(peak_signal_noise_ratio(pre_image, dri_image))
            SSIM.append(structural_similarity(pre_image, dri_image, multichannel=True, channel_axis=2))

        if not os.path.exists(video_save_dir):
            os.makedirs(video_save_dir)
        for idx, frame in enumerate(output_frames):
            # io.imsave(os.path.join(video_save_dir, "%s.png" % str(idx).zfill(8)), frame)
            cv2.imwrite(os.path.join(video_save_dir, "%s.png" % str(idx).zfill(8)), frame[:, :, ::-1])
    loss_str = f"Reconstruction_copy loss: {np.mean(L1)} \n \
        MSE loss: {(np.mean(MSE))} \n \
        PSNR loss: {(np.mean(PSNR))} \n \
        SSIM loss: {(np.mean(SSIM))}"

    print(loss_str)
    with open(os.path.join(save_dir, "reconstruction.txt"), "w") as f:
        f.write(loss_str)
    
    print("Reconstruction done!")


@torch.no_grad()
def reenactment(hp, g_models, source_dir, img_dir, img_size, save_dir):
    source_paths = sorted(glob.glob(os.path.join(source_dir, "*")))
    video_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    for ind, video_path in enumerate(video_paths):
        video_save_dir = os.path.join(save_dir, os.path.basename(video_path))
        img_paths = sorted(glob.glob(os.path.join(video_path, "*.png")))
        s = img_as_float32(io.imread(source_paths[ind]))[:, :, :3]
        s = np.array(s, dtype="float32").transpose((2, 0, 1))
        s = torch.from_numpy(s).unsqueeze(0).cuda()
        s = F.interpolate(s, size=(img_size, img_size), mode="bilinear", align_corners=False)
        fs = g_models["afe"](s)
        kp_c = g_models["ckd"](s)
        _, _, _, t_s, scale_s = g_models["hpe_ede"](s)
        yaw_s, pitch_s, roll_s = hp(F.interpolate(apply_imagenet_normalization(s), size=(224, 224)))
        delta_s, _, _, _, _ = g_models["efe"](s, None, kp_c)
        kp_s, Rs = transform_kp(kp_c+delta_s, yaw_s, pitch_s, roll_s, t_s, scale_s)
        output_frames = []
        for idx, dri in tqdm(enumerate(img_paths), total=len(img_paths), desc=f"{ind}/{len(video_paths)}:"):
            dri_image = io.imread(dri)[:, :, :3]
            # dri_image = cv2.cvtColor(cv2.imread(dri)[:, :, :3], cv2.COLOR_BGR2RGB)
            img = img_as_float32(dri_image)
            img = np.array(img, dtype="float32").transpose((2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            img = F.interpolate(img, size=(img_size, img_size), mode="bilinear", align_corners=False)
            _, _, _, t_d, scale_d = g_models["hpe_ede"](img)
            # kp_c_d = g_models["ckd"](img)
            yaw_d, pitch_d, roll_d = hp(F.interpolate(apply_imagenet_normalization(img), size=(224, 224)))
            delta_d, _, _, _, _ = g_models["efe"](img, None, kp_c)
            kp_d, Rd = transform_kp(kp_c+delta_d, yaw_d, pitch_d, roll_d, t_d, scale_d)
            deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d, Rs, Rd)
            generated_d, _, _ = g_models["generator"](fs, deformation, occlusion)
            
            generated_d = generated_d.squeeze(0).data.cpu().numpy()
            generated_d = np.transpose(generated_d, [1, 2, 0])
            generated_d = generated_d.clip(0, 1)
            pre_image = (255 * generated_d).astype(np.uint8)
            output_frames.append(pre_image)

        if not os.path.exists(video_save_dir):
            os.makedirs(video_save_dir)
        for idx, frame in enumerate(output_frames):
            # io.imsave(os.path.join(video_save_dir, "%s.png" % str(idx).zfill(8)), frame)
            cv2.imwrite(os.path.join(video_save_dir, "%s.png" % str(idx).zfill(8)), frame[:, :, ::-1])
    
    print("Reenactment done!")

@torch.no_grad()
def reenactment_multi(hp, g_models, source_dir, ree_dir, img_size, save_dir):
    video_paths = sorted(glob.glob(os.path.join(ree_dir, "*")))
    source_paths = sorted(glob.glob(os.path.join(source_dir, "*.png")))

    for ind, source_path in enumerate(source_paths):
        s = img_as_float32(io.imread(source_path))[:, :, :3]
        s = np.array(s, dtype="float32").transpose((2, 0, 1))
        s = torch.from_numpy(s).unsqueeze(0).cuda()
        s = F.interpolate(s, size=(img_size, img_size), mode="bilinear", align_corners=False)
        fs = g_models["afe"](s)
        kp_c = g_models["ckd"](s)
        _, _, _, t_s, scale_s = g_models["hpe_ede"](s)
        yaw_s, pitch_s, roll_s = hp(F.interpolate(apply_imagenet_normalization(s), size=(224, 224)))
        delta_s, _, _, _, _ = g_models["efe"](s, None, kp_c)
        kp_s, Rs = transform_kp(kp_c+delta_s, yaw_s, pitch_s, roll_s, t_s, scale_s)
        output_frames = []
        for vid, video_path in enumerate(video_paths):
            if os.path.isdir(video_path):
                img_paths = sorted(glob.glob(os.path.join(video_path, "*.png")))
                for idx, dri in tqdm(enumerate(img_paths), total=len(img_paths), desc=f"{ind}/{len(source_paths)}_{vid}/{len(video_paths)}"):            
                    dri_image = io.imread(dri)[:, :, :3]
                    # dri_image = cv2.cvtColor(cv2.imread(dri)[:, :, :3], cv2.COLOR_BGR2RGB)
                    img = img_as_float32(dri_image)
                    img = np.array(img, dtype="float32").transpose((2, 0, 1))
                    img = torch.from_numpy(img).unsqueeze(0).cuda()
                    img = F.interpolate(img, size=(img_size, img_size), mode="bilinear", align_corners=False)
                    _, _, _, t_d, scale_d = g_models["hpe_ede"](img)
                    kp_c_d = g_models["ckd"](img)
                    yaw_d, pitch_d, roll_d = hp(F.interpolate(apply_imagenet_normalization(img), size=(224, 224)))
                    delta_d, _, _, _, _ = g_models["efe"](img, None, kp_c_d)
                    kp_d, Rd = transform_kp(kp_c+delta_d, yaw_d, pitch_d, roll_d, t_d, scale_d)
                    deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d, Rs, Rd)
                    generated_d, _, _ = g_models["generator"](fs, deformation, occlusion)            
                    generated_d = generated_d.squeeze(0).data.cpu().numpy()
                    generated_d = np.transpose(generated_d, [1, 2, 0])
                    generated_d = generated_d.clip(0, 1)
                    pre_image = (255 * generated_d).astype(np.uint8)
                    output_frames.append(pre_image)
            else:
                video_array = np.array(mimread(video_path))[..., :3]
                for idx, dri_image in tqdm(enumerate(video_array), total=len(video_array), desc=f"{ind}/{len(source_paths)}_{vid}/{len(video_paths)}"):
                    img = img_as_float32(dri_image)
                    img = np.array(img, dtype="float32").transpose((2, 0, 1))
                    img = torch.from_numpy(img).unsqueeze(0).cuda()
                    img = F.interpolate(img, size=(img_size, img_size), mode="bilinear", align_corners=False)
                    _, _, _, t_d, scale_d = g_models["hpe_ede"](img)
                    kp_c_d = g_models["ckd"](img)
                    yaw_d, pitch_d, roll_d = hp(F.interpolate(apply_imagenet_normalization(img), size=(224, 224)))
                    delta_d, _, _, _, _ = g_models["efe"](img, None, kp_c_d)
                    kp_d, Rd = transform_kp(kp_c+delta_d, yaw_d, pitch_d, roll_d, t_d, scale_d)
                    deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d, Rs, Rd)
                    generated_d, _, _ = g_models["generator"](fs, deformation, occlusion)            
                    generated_d = generated_d.squeeze(0).data.cpu().numpy()
                    generated_d = np.transpose(generated_d, [1, 2, 0])
                    generated_d = generated_d.clip(0, 1)
                    pre_image = (255 * generated_d).astype(np.uint8)
                    output_frames.append(pre_image)

            video_save_dir = os.path.join(save_dir, f's_{os.path.basename(source_path)}-d_{os.path.basename(video_path)}')
            # if not os.path.exists(video_save_dir):
            #     os.makedirs(video_save_dir)
            for idx, frame in enumerate(output_frames):
                # io.imsave(os.path.join(video_save_dir, "%s.png" % str(idx).zfill(8)), frame)
                cv2.imwrite(video_save_dir+ f"-{idx}.png", frame[:, :, ::-1])

    print("save path:", save_dir)    
    print("Reenactment done!")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = argparse.ArgumentParser("face-vae")
    parser.add_argument("--hp_pickle_path", default="hopenet_robust_alpha1.pkl", help="hp_pickle_path")
    parser.add_argument("--ckp_dir", default="ckp_1644_mainv9finalv3-lml-dls-newgenmodel-mask-sslr-c100-lp-ld", help="checkpoint_dir")
    parser.add_argument("--ckp", default=221, help="checkpoint_epoch")
    # parser.add_argument("--img_dir", default="/home/lh/repo3/lh/repo/datasets/face-video-preprocessing/vox-png/test", help="img_dir")
    parser.add_argument("--img_size", default=256, help="img_size")
    parser.add_argument("--batch_size", default=1, help="batch_size")
    parser.add_argument("--save_dir", default="/data1/datasets/vox1/metric/tiaotu0424", help="save_dir")
    # parser.add_argument("--save_dir", default="demo/out", help="save_dir")

    parser.add_argument("--save_dir_name", default="newgen-mask-sslr-c100", help="save_dir")
    parser.add_argument("--source_dir", default="/data1/datasets/vox1/metric/tiaotu0424/s2", help="source_img_dir")
    parser.add_argument("--ree_dir", default="/data1/datasets/vox1/metric/tiaotu0424/d2-png", help="ree_img_dir")
    # parser.add_argument("--ree_dir", default="/data/repo/code/lh/2.faceanimation/face-vae/demo/dri", help="ree_img_dir")

    parser.add_argument("--flag", default="reconstruction", 
                        choices=["reconstruction", "reenactment"], 
                        help="reconstruction or reenactment")

    args = parser.parse_args()
    hp, g_models = init_model(args)

    if args.flag == "reconstruction":
        rec_save_dir = os.path.join(args.save_dir, 'rec', args.save_dir_name + "_" + str(args.ckp))
        reconstruction(hp, g_models, args.ree_dir, args.img_size, rec_save_dir)
    elif args.flag == "reenactment":
        ree_save_dir = os.path.join(args.save_dir, 'ree', args.save_dir_name + "_" + str(args.ckp) + "_ree")
        os.makedirs(ree_save_dir)
        reenactment(hp, g_models, args.source_dir, args.ree_dir, args.img_size, ree_save_dir)
    
