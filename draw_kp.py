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
from skimage import io, img_as_float32
from utils import transform_kp, transform_kp_with_new_pose
from logger import Visualizer
import torchvision
import math
from utils import apply_imagenet_normalization


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
# imageio.plugins.freeimage.download()
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
    
    source_paths = []
    for dirpath, dirnames, filenames in os.walk(args.source, followlinks=True):
        print(dirpath)
        # continue
        pattern = re.compile(r'^[^\.].*\.(jpg|png|webp|jpeg|JPG)$')
        for filename in filenames:
            # print(filename)
            if not pattern.match(filename): continue
            img_path = os.path.join(dirpath, filename)
            # print(filename)
            print(img_path)
            source_paths.append(img_path)
            # continue
    
    driving_paths = []      
    for dirpath, dirnames, filenames in os.walk(args.driving, followlinks=True):
        print(dirpath)
        # continue
        pattern = re.compile(r'^[^\.].*\.(jpg|png|webp|jpeg|JPG)$')
        if dirpath[-3:]=='out': continue
        for filename in filenames:
            # print(filename)
            if not pattern.match(filename): continue
            img_path = os.path.join(dirpath, filename)
            # print(filename)
            print(img_path)
            # continue
            driving_paths.append(img_path)
            

    print(source_paths)
    print(driving_paths)
    
    vs = Visualizer()

    for idx, src in enumerate(source_paths):
        s = img_as_float32(io.imread(src))[:, :, :3]
        s = np.array(s, dtype="float32").transpose((2, 0, 1))
        s = torch.from_numpy(s).cuda().unsqueeze(0)
        s = F.interpolate(s, size=(256, 256))
        fs = g_models["afe"](s)
        kp_c = g_models["ckd"](s)
        _, _, _, t, scale = g_models["hpe_ede"](s)
        with torch.no_grad():
            hp.eval()
            yaw, pitch, roll = hp(F.interpolate(apply_imagenet_normalization(s), size=(224, 224)))
        # kp_s, Rs = transform_kp(kp_c, yaw, pitch, roll, t, scale)
        # kp_s, _, _, _, _ = g_models["efe"](s, None, kp_s)
        delta_s, _, _, _, _ = g_models["efe"](s, None, kp_c)
        kp_s, Rs = transform_kp(kp_c+delta_s, yaw, pitch, roll, t, scale)
        
        for dri in driving_paths:
            img = img_as_float32(io.imread(dri))[:, :, :3]
            img = np.array(img, dtype="float32").transpose((2, 0, 1))
            img = torch.from_numpy(img).cuda().unsqueeze(0)
            _, _, _, t, scale = g_models["hpe_ede"](img)
            with torch.no_grad():
                hp.eval()
                yaw, pitch, roll = hp(F.interpolate(apply_imagenet_normalization(img), size=(224, 224)))
            
            kp_c_d = g_models["ckd"](img)
            # delta = delta
            delta_d, _, _, _, _ = g_models["efe"](img, None, kp_c_d)
            kp_d1 = kp_c
            kp_d2 = kp_c + delta_d
            kp_d3, Rd3 = transform_kp(kp_c + delta_d, yaw*0, pitch*0, roll*0, t*0, scale*0.85)
            kp_d4, Rd4= transform_kp(kp_c + delta_d, yaw, pitch, roll, t*0, scale)
            kp_d5, Rd5 = transform_kp(kp_c + delta_d, yaw, pitch, roll, t, scale)
            kp_d6, Rd6 = transform_kp(kp_c, yaw, pitch, roll, t, scale)
            kp_d7, Rd7 = kp_d6 + transform_kp(delta_d, yaw, pitch, roll, t*0, scale)[0], Rd6
            kp_d8, Rd8 = transform_kp(kp_c + delta_d, yaw, pitch, roll, t, scale*0.8)

            # deformation, occlusion, mask = g_models["mfe"](fs, kp_s, kp_d8, Rs, Rd)
            # generated_d = g_models["generator"](fs, deformation, occlusion)
            deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d1, Rs, Rd3)
            generated_d1, _, _ = g_models["generator"](fs, deformation, occlusion)
            deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d2, Rs, Rd3)
            generated_d2, _, _ = g_models["generator"](fs, deformation, occlusion)
            deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d3, Rs, Rd3)
            generated_d3, _, _ = g_models["generator"](fs, deformation, occlusion)	
            deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d4, Rs, Rd4)
            generated_d4, _, _ = g_models["generator"](fs, deformation, occlusion)
            deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d5, Rs, Rd5)
            generated_d5, _, _ = g_models["generator"](fs, deformation, occlusion)
            deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d6, Rs, Rd6)
            generated_d6, _, _ = g_models["generator"](fs, deformation, occlusion)
            deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d7, Rs, Rd7)
            generated_d7, _, _ = g_models["generator"](fs, deformation, occlusion)
            deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d8, Rs, Rd8)
            generated_d8, _, _ = g_models["generator"](fs, deformation, occlusion)

            s_np = s.data.cpu()
            kp_s_np = kp_s.data.cpu().numpy()[:, :, :2]
            s_np = np.transpose(s_np, [0, 2, 3, 1])
            img_np = img.data.cpu()
            # generated_d_np = generated_d.data.cpu()

            kp_d1_np = kp_d1.data.cpu().numpy()[:, :, :2]
            kp_d2_np = kp_d2.data.cpu().numpy()[:, :, :2]
            kp_d3_np = kp_d3.data.cpu().numpy()[:, :, :2]
            kp_d4_np = kp_d4.data.cpu().numpy()[:, :, :2]
            kp_d5_np = kp_d5.data.cpu().numpy()[:, :, :2]
            kp_d6_np = kp_d6.data.cpu().numpy()[:, :, :2]
            kp_d7_np = kp_d7.data.cpu().numpy()[:, :, :2]
            kp_d8_np = kp_d8.data.cpu().numpy()[:, :, :2]

            img_np = np.transpose(img_np, [0, 2, 3, 1])
            # generated_d_np = np.transpose(generated_d_np, [0, 2, 3, 1])
            img_with_kp = [(s_np, kp_s_np), (img_np, kp_d1_np), (img_np, kp_d2_np), (img_np, kp_d3_np), (img_np, kp_d4_np), (img_np, kp_d5_np), (img_np, kp_d6_np), (img_np, kp_d7_np), (img_np, kp_d8_np)]
            img_with_kp = vs.create_image_grid(*img_with_kp)
            img_with_kp = img_with_kp.clip(0, 1)
            img_with_kp = (255 * img_with_kp).astype(np.uint8)
            # imageio.mimsave(args.output, output_frames)
            os.makedirs(os.path.dirname(dri) + '_kp_outv9fv3_lml_dls-newgen-mask-largeW-214', exist_ok=True)
            imageio.imwrite(os.path.dirname(dri) + '_kp_outv9fv3_lml_dls-newgen-mask-largeW-214' + '/' + f'{idx}_'+os.path.basename(dri) , img_with_kp)
        
        # os.makedirs(dirpath+'out', exist_ok=True)
        # out_path = os.path.join(dirpath+'out', filename)
            generated_d1_np = generated_d1.data.cpu().numpy().transpose([0, 2, 3, 1])
            generated_d2_np = generated_d2.data.cpu().numpy().transpose([0, 2, 3, 1])
            generated_d3_np = generated_d3.data.cpu().numpy().transpose([0, 2, 3, 1])
            generated_d4_np = generated_d4.data.cpu().numpy().transpose([0, 2, 3, 1])
            generated_d5_np = generated_d5.data.cpu().numpy().transpose([0, 2, 3, 1])
            generated_d6_np = generated_d6.data.cpu().numpy().transpose([0, 2, 3, 1])
            generated_d7_np = generated_d7.data.cpu().numpy().transpose([0, 2, 3, 1])
            generated_d8_np = generated_d8.data.cpu().numpy().transpose([0, 2, 3, 1])
            img_d = [(s_np, kp_s_np),
                    (generated_d1_np, kp_d1_np), (generated_d2_np, kp_d2_np), (generated_d3_np, kp_d3_np), 
                    (generated_d4_np, kp_d4_np), (generated_d5_np, kp_d5_np), (generated_d6_np, kp_d6_np),
                    (generated_d7_np, kp_d7_np), (generated_d8_np, kp_d8_np)]

            img_d = vs.create_image_grid(*img_d)
            img_d = (255 * img_d).astype(np.uint8)
            imageio.imwrite(os.path.dirname(dri) + '_kp_outv9fv3_lml_dls-newgen-mask-largeW-214' + '/' + f'{idx}_d_'+os.path.basename(dri), img_d)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = argparse.ArgumentParser(description="face-vid2vid")

    def str2bool(s):
        return s.lower().startswith("t")

    parser.add_argument("--ckp_dir", type=str, default="ckp_1644_mainv9finalv3-lml-dls-newgenmodel-mask-slr-sc-lp-ld", help="Checkpoint dir")
    parser.add_argument("--output", type=str, default="output.gif", help="Output video")
    parser.add_argument("--ckp", type=int, default=214, help="Checkpoint epoch")
    parser.add_argument("--source", type=str, default="demo/s", help="Source image, f for face frontalization, r for reconstruction")
    parser.add_argument("--driving", type=str, default='demo/out/rec/newgen-mask-lw_181/ENGLAND_00035.mp4', help="Driving dir")
    parser.add_argument("--num_frames", type=int, default=90, help="Number of frames")

    args = parser.parse_args()
    # eval(args)
    demo(args)
    
