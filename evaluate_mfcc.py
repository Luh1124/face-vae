import argparse
from models import AFE, CKD, HPE_EDE, MFE, Generator
from models import EFE_6 as EFE
from models_vae import VAE, AudioVAE

import numpy as np
import torch
import torch.nn.functional as F
import imageio
import os
from skimage import io, img_as_float32
from utils import transform_kp, transform_kp_with_new_pose
from scipy.spatial import ConvexHull


import argparse
from models import EFE_6 as EFE
from models import AFE, CKD, HPE_EDE, MFE, Generator
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
    vae = VAE().cuda().eval()
    vae_ckp = torch.load("ckp_video_vae/00000059-checkpoint.pth.tar", map_location=torch.device("cpu"))

    audiovae = AudioVAE().cuda().eval()
    audiovae_ckp = torch.load("ckp_audio_vae/00000009-checkpoint.pth.tar", map_location=torch.device("cpu"))
    

    for k, v in g_models.items():
        v.cuda()
        v.load_state_dict(checkpoint[k])
        v.eval()
    vae.load_state_dict(vae_ckp["vae"])
    audiovae.load_state_dict(audiovae_ckp["audiovae"])

    source_paths = []
    source_paths.append(args.source)
    driving_paths = []
    driving_paths.append(args.driving)
    
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

        img_d_array = []
        for dri in driving_paths:
            frames = os.listdir(dri)
            num_frames = len(frames)
            # frame_idx = np.sort(np.random.choice(range(4, num_frames-4), replace=True, size=1))
            video_array = [img_as_float32(io.imread(os.path.join(dri, str(frames[idx]))))[:, :, :3] for idx in range(4,num_frames-4)]

            mfccs = np.load(args.audio_path)[4:num_frames-4]

            for frame_idx, (mfcc, img) in enumerate(zip(video_array, mfccs)):
                img = np.array(img, dtype="float32").transpose((2, 0, 1))
                img = torch.from_numpy(img).cuda().unsqueeze(0)

                mfcc = np.array(mfccs[0], dtype='float32')
                mfcc = (mfcc-np.min(mfcc))/(np.max(mfcc)-np.min(mfcc))
                mfcc = torch.from_numpy(mfcc).unsqueeze(1).cuda().unsqueeze(0)

                _, _, _, t, scale = g_models["hpe_ede"](img)
                with torch.no_grad():
                    hp.eval()
                    yaw, pitch, roll = hp(F.interpolate(apply_imagenet_normalization(img), size=(224, 224)))
                    kp_c_d = g_models["ckd"](img)

                    # delta = delta
                    # delta_d, _, _, _, _ = g_models["efe"](img, None, kp_c_d)
                
                # feature_s = g_models["efe"].encode(s)
                feature_d_ori = g_models["efe"].encode(img)

                
                feature_d_10 = feature_d_ori * 10.
                feature_d_vae = vae.generate(feature_d_10.flatten(1)) / 10.

                audio_z = audiovae(mfcc)[1]
                feature_d_audio = vae.decode(audio_z)


                # feature_d = vae.decode(feature_s_z/2. + feature_d_z/2.).view(-1, 16, 4, 4) / 10.

                delta_d_ori = g_models["efe"].decode(feature_d_ori, kp_c_d)
                delta_d_vae = g_models["efe"].decode(feature_d_vae, kp_c_d)
                delta_d_audio = g_models["efe"].decode(feature_d_audio, kp_c_d)
                
                kp_d1, Rd1 = transform_kp(kp_c + delta_d_ori, yaw, pitch, roll, t, scale)
                kp_d2, Rd2= transform_kp(kp_c + delta_d_vae, yaw, pitch, roll, t, scale)
                kp_d3, Rd3 = transform_kp(kp_c + delta_d_audio, yaw, pitch, roll, t, scale)

                # kp_d = g_models["efe"].decoder(feature_d_0/10., kp_d_old)

                # kp_d = normalize_kp(kp_source=kp_s, kp_driving=kp_d,
                #             kp_driving_initial=None, use_relative_movement=False,
                #             adapt_movement_scale=True)
                
                deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d1, Rs, Rd1)
                generated_d1 = g_models["generator"](fs, deformation, occlusion)
                deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d2, Rs, Rd2)
                generated_d2 = g_models["generator"](fs, deformation, occlusion)
                deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d3, Rs, Rd3)
                generated_d3 = g_models["generator"](fs, deformation, occlusion)	
            
                s_np = s.data.cpu()
                kp_s_np = kp_s.data.cpu().numpy()[:, :, :2]
                s_np = np.transpose(s_np, [0, 2, 3, 1])
                img_np = img.data.cpu()
                img_np = np.transpose(img_np, [0, 2, 3, 1])

                kp_d1_np = kp_d1.data.cpu().numpy()[:, :, :2]
                kp_d2_np = kp_d2.data.cpu().numpy()[:, :, :2]
                kp_d3_np = kp_d3.data.cpu().numpy()[:, :, :2]

                generated_d1_np = generated_d1.data.cpu().numpy().transpose([0, 2, 3, 1])
                generated_d2_np = generated_d2.data.cpu().numpy().transpose([0, 2, 3, 1])
                generated_d3_np = generated_d3.data.cpu().numpy().transpose([0, 2, 3, 1])

                img_d = [(s_np, kp_s_np),
                    (generated_d1_np, kp_d1_np), (generated_d2_np, kp_d2_np), (generated_d3_np, kp_d3_np)]

                img_d = vs.create_image_grid(*img_d)
                img_d = (255 * img_d).astype(np.uint8)

                imageio.imwrite(args.output + '/' + f'{frame_idx}_d_'+os.path.basename(dri), img_d)
                img_d_array.append(img_d)
            
            imageio.mimsave(args.output + '/output.gif', img_d_array, fps=25)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face-vae")

    def str2bool(s):
        return s.lower().startswith("t")

    parser.add_argument("--ckp_dir", type=str, default="ckp_1644_main13v2_hid", help="Visualization dir")
    parser.add_argument("--ckp", type=int, default=119, help="Checkpoint epoch")

    parser.add_argument("--source", type=str, default="dvideo/ABOUT_00042.mp4/00000004.png", help="Source image, f for face frontalization, r for reconstruction")
    parser.add_argument("--driving", type=str, default='dvideo/ABOUT_00042.mp4', help="Driving dir")
    parser.add_argument("--audio_path", type=str, default='dmfcc/ABOUT_00042.npy', help="Driving dir")

    parser.add_argument("--output", type=str, default="audio_output", help="Output video")

    args = parser.parse_args()
    # eval(args)
    demo(args)
    
