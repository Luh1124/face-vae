import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torch import nn
from utils import apply_imagenet_normalization, apply_vggface_normalization
import cv2

def l1(x, y):
    return torch.abs(x-y)


def l2(x, y):
    return torch.pow((x-y), 2)


@torch.jit.script
def fuse_math_min_mean_pos(x):
    r"""Fuse operation min mean for hinge loss computation of positive
    samples"""
    minval = torch.min(x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss


@torch.jit.script
def fuse_math_min_mean_neg(x):
    r"""Fuse operation min mean for hinge loss computation of negative
    samples"""
    minval = torch.min(-x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss


class _PerceptualNetwork(nn.Module):
    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        self.network = network.cuda()
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                output[layer_name] = x
        return output


def _vgg19(layers):
    network = torchvision.models.vgg19()
    state_dict = torch.utils.model_zoo.load_url(
        "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth", map_location=torch.device("cpu"), progress=True
    )
    network.load_state_dict(state_dict)
    network = network.features
    layer_name_mapping = {
        1: "relu_1_1",
        3: "relu_1_2",
        6: "relu_2_1",
        8: "relu_2_2",
        11: "relu_3_1",
        13: "relu_3_2",
        15: "relu_3_3",
        17: "relu_3_4",
        20: "relu_4_1",
        22: "relu_4_2",
        24: "relu_4_3",
        26: "relu_4_4",
        29: "relu_5_1",
    }
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg_face(layers):
    network = torchvision.models.vgg16(num_classes=2622)
    state_dict = torch.utils.model_zoo.load_url(
        "http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/" "vgg_face_dag.pth", map_location=torch.device("cpu"), progress=True
    )
    feature_layer_name_mapping = {
        0: "conv1_1",
        2: "conv1_2",
        5: "conv2_1",
        7: "conv2_2",
        10: "conv3_1",
        12: "conv3_2",
        14: "conv3_3",
        17: "conv4_1",
        19: "conv4_2",
        21: "conv4_3",
        24: "conv5_1",
        26: "conv5_2",
        28: "conv5_3",
    }
    new_state_dict = {}
    for k, v in feature_layer_name_mapping.items():
        new_state_dict["features." + str(k) + ".weight"] = state_dict[v + ".weight"]
        new_state_dict["features." + str(k) + ".bias"] = state_dict[v + ".bias"]
    classifier_layer_name_mapping = {0: "fc6", 3: "fc7", 6: "fc8"}
    for k, v in classifier_layer_name_mapping.items():
        new_state_dict["classifier." + str(k) + ".weight"] = state_dict[v + ".weight"]
        new_state_dict["classifier." + str(k) + ".bias"] = state_dict[v + ".bias"]
    network.load_state_dict(new_state_dict)
    layer_name_mapping = {
        1: "relu_1_1",
        3: "relu_1_2",
        6: "relu_2_1",
        8: "relu_2_2",
        11: "relu_3_1",
        13: "relu_3_2",
        15: "relu_3_3",
        18: "relu_4_1",
        20: "relu_4_2",
        22: "relu_4_3",
        25: "relu_5_1",
    }
    return _PerceptualNetwork(network.features, layer_name_mapping, layers)


class PerceptualLoss(nn.Module):
    def __init__(self, layers_weight={"relu_1_1": 0.03125, "relu_2_1": 0.0625, "relu_3_1": 0.125, "relu_4_1": 0.25, "relu_5_1": 1.0}, n_scale=3):
        super().__init__()
        self.vgg19 = _vgg19(layers_weight.keys())
        self.vggface = _vgg_face(layers_weight.keys())
        self.criterion = nn.L1Loss()
        self.layers_weight, self.n_scale = layers_weight, n_scale

    def forward(self, input, target):
        self.vgg19.eval()
        self.vggface.eval()
        loss = 0
        loss += self.criterion(input, target)
        features_vggface_input = self.vggface(apply_vggface_normalization(input))
        features_vggface_target = self.vggface(apply_vggface_normalization(target))
        input = apply_imagenet_normalization(input)
        target = apply_imagenet_normalization(target)
        features_vgg19_input = self.vgg19(input)
        features_vgg19_target = self.vgg19(target)
        for layer, weight in self.layers_weight.items():
            loss += weight * self.criterion(features_vggface_input[layer], features_vggface_target[layer].detach()) / 255
            loss += weight * self.criterion(features_vgg19_input[layer], features_vgg19_target[layer].detach())
        for i in range(self.n_scale):
            input = F.interpolate(input, mode="bilinear", scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
            target = F.interpolate(target, mode="bilinear", scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
            features_vgg19_input = self.vgg19(input)
            features_vgg19_target = self.vgg19(target)
            loss += weight * self.criterion(features_vgg19_input[layer], features_vgg19_target[layer].detach())
        return loss


class GANLoss(nn.Module):
    # Update generator: gan_loss(fake_output, True, False) + other losses
    # Update discriminator: gan_loss(fake_output(detached), False, True) + gan_loss(real_output, True, True)
    def __init__(self):
        super().__init__()

    def forward(self, dis_output, t_real, dis_update=True):
        r"""GAN loss computation.
        Args:
            dis_output (tensor or list of tensors): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise
                uses the fake label as target.
            dis_update (bool): If ``True``, the loss will be used to update the
                discriminator, otherwise the generator.
        Returns:
            loss (tensor): Loss value.
        """

        if dis_update:
            if t_real:
                loss = fuse_math_min_mean_pos(dis_output)
            else:
                loss = fuse_math_min_mean_neg(dis_output)
        else:
            loss = -torch.mean(dis_output)
        return loss


class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, fake_features, real_features):
        num_d = len(fake_features)
        dis_weight = 1.0 / num_d
        loss = fake_features[0][0].new_tensor(0)
        for i in range(num_d):
            for j in range(len(fake_features[i])):
                tmp_loss = self.criterion(fake_features[i][j], real_features[i][j].detach())
                loss += dis_weight * tmp_loss
        return loss


class EquivarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, kp_d, reverse_kp):
        loss = self.criterion(kp_d[:, :, :2], reverse_kp)
        return loss


class KeypointPriorLoss(nn.Module):
    def __init__(self, Dt=0.1, zt=0.33):
        super().__init__()
        self.Dt, self.zt = Dt, zt

    def forward(self, kp_d):
        # use distance matrix to avoid loop
        dist_mat = torch.cdist(kp_d, kp_d).square()
        loss = (
            torch.max(0 * dist_mat, self.Dt - dist_mat).sum((1, 2)).mean()
            + torch.abs(kp_d[:, :, 2].mean(1) - self.zt).mean()
            - kp_d.shape[1] * self.Dt
        )
        return loss

class HeadPoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, yaw, pitch, roll, real_yaw, real_pitch, real_roll):
        loss = (self.criterion(yaw, real_yaw.detach()) + self.criterion(pitch, real_pitch.detach()) + self.criterion(roll, real_roll.detach())) / 3
        return loss / np.pi * 180


class DeformationPriorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, delta_d):
        loss = delta_d.abs().mean()
        return loss


class ContrastiveLoss_linear(nn.Module):
    def __init__(self, in_dim=256, hid_1_dim=256, out_1_dim=256, hid_2_dim=256, out_2_dim=256, mode="direction") -> None:
        super(ContrastiveLoss_linear, self).__init__()
        self.criterion = nn.CosineSimilarity(dim=1)
        self.mode = mode

        if self.mode != "direction":
            # build a 3-layer projector
            self.projection = nn.Sequential(nn.Linear(in_dim, hid_1_dim, bias=False),
                                        nn.BatchNorm1d(hid_1_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(hid_1_dim, hid_1_dim, bias=False),
                                        nn.BatchNorm1d(hid_1_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(hid_1_dim, out_1_dim, bias=True),
                                        nn.BatchNorm1d(out_1_dim, affine=False)) # output layer
            self.projection[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

            # build a 2-layer predictor
            self.predictor = nn.Sequential(nn.Linear(out_1_dim, hid_2_dim, bias=False),
                                        nn.BatchNorm1d(hid_2_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(hid_2_dim, out_2_dim)) # output layer
    def forward(self, f1, f2):
        # loss = 0.0
        f1 = f1.view(f1.shape[0], -1)
        f2 = f2.view(f2.shape[0], -1)
        if self.mode != "direction":
            z1 = self.projection(f1) # NxC
            z2 = self.projection(f2) # NxC
            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC
            loss = 1 - (self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
        else:
            loss = 1 - self.criterion(f1,f2).mean()
        return loss


class ContrastiveLoss_conv(nn.Module):
    def __init__(self, in_dim=256, hid_1_dim=128, out_1_dim=128, hid_2_dim=64, out_2_dim=3, mode="direction") -> None:
        super(ContrastiveLoss_conv, self).__init__()
        from taming.modules.losses.lpips import LPIPS
        
        self.criterion = LPIPS().cuda().eval()
        # self.criterion = nn.L1Loss()
        self.mode = mode

        if self.mode != "direction":
            # build a 3-layer projector
            self.projection = nn.Sequential(nn.Conv2d(in_dim, hid_1_dim, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(hid_1_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Conv2d(hid_1_dim, hid_1_dim, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(hid_1_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Conv2d(hid_1_dim, out_1_dim, kernel_size=3, padding=1, bias=True),
                                        nn.BatchNorm2d(out_1_dim, affine=False)) # output layer
            self.projection[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

            # build a 2-layer predictor
            self.predictor = nn.Sequential(nn.Conv2d(out_1_dim, hid_2_dim, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(hid_2_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Conv2d(hid_2_dim, out_2_dim, kernel_size=3, padding=1)) # output layer
        else:
            self.projection = nn.Conv2d(256, 3, kernel_size=1, padding=0) # output layer
            # self.weight = (torch.ones(3, 256, 1, 1) / 256)

    def forward(self, f1, f2):
        # loss = 0.0
        if self.mode != "direction":
            z1 = self.projection(f1) # NxC
            z2 = self.projection(f2) # NxC
            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC
            loss = 1 - (self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
        else:
            # z1= F.conv2d(f1, weight=self.weight)
            # z2= F.conv2d(f2, weight=self.weight)

            z1 = self.projection(f1)
            z2 = self.projection(f2)
            loss = self.criterion(z1,z2).mean()
        return loss


class ContrastiveLoss_conv2(nn.Module):
    def __init__(self, in_dim=256, out_dim=128, dim_linear=512, mode="direction") -> None:
        super(ContrastiveLoss_conv2, self).__init__()
        self.criterion = nn.CosineSimilarity(dim=1)
        self.mode = mode

        if self.mode != "direction":
            # build a 3-layer projector
            # self.projection = nn.Sequential(nn.Conv2d(in_dim, hid_1_dim, kernel_size=3, padding=1, bias=False),
            #                             nn.BatchNorm2d(hid_1_dim),
            #                             nn.ReLU(inplace=True), # first layer
            #                             nn.Conv2d(hid_1_dim, hid_1_dim, kernel_size=3, padding=1, bias=False),
            #                             nn.BatchNorm2d(hid_1_dim),
            #                             nn.ReLU(inplace=True), # second layer
            #                             nn.Conv2d(hid_1_dim, out_1_dim, kernel_size=3, padding=1, bias=True),
            #                             nn.BatchNorm2d(out_1_dim, affine=False)) # output layer
            # self.projection[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

            self.projection = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=True),
                                        nn.BatchNorm2d(out_dim, affine=False)) # output layer
            self.projection[0].bias.requires_grad = False # hack: not use bias as it is followed by BN

            # build a 2-layer predictor
            # self.predictor = nn.Sequential(nn.Conv2d(out_1_dim, hid_2_dim, kernel_size=3, padding=1, bias=False),
            #                             nn.BatchNorm2d(hid_2_dim),
            #                             nn.ReLU(inplace=True), # hidden layer
            #                             nn.Conv2d(hid_2_dim, out_2_dim, kernel_size=3, padding=1)) # output layer

            self.predictor = nn.Sequential(nn.Linear(dim_linear, dim_linear, bias=False),
                                        nn.BatchNorm1d(dim_linear),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(dim_linear, dim_linear)) # output layer
        else:
            self.projection = nn.Conv2d(256, 3, kernel_size=1, padding=0) # output layer
            # self.weight = (torch.ones(3, 256, 1, 1) / 256)

    def forward(self, f1, f2):
        # loss = 0.0
        if self.mode != "direction":
            z1 = self.projection(f1) # NxC
            z2 = self.projection(f2) # NxC
            z1 = z1.view(z1.shape[0], -1)
            z2 = z2.view(z2.shape[0], -1)
            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC
            loss = 1 - (self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
        else:
            # z1= F.conv2d(f1, weight=self.weight)
            # z2= F.conv2d(f2, weight=self.weight)

            z1 = self.projection(f1)
            z2 = self.projection(f2)
            loss = self.criterion(z1,z2).mean()
        return loss


class KLDivergenceLoss(nn.Module):
    def __init__(self) -> None:
        super(KLDivergenceLoss, self).__init__()

    def forward(self, kl):
        mu = kl[0]
        logstd = kl[1]
        loss = torch.mean(-0.5 - logstd + 0.5 * mu ** 2 + 0.5 * torch.exp(2 * logstd), dim=-1).mean()
        return loss


class ReconLoss(nn.Module):
    def __init__(self) -> None:
        super(ReconLoss, self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(self, Rec):
        loss = self.mseloss(Rec[0], Rec[1])
        return loss


class IdLoss(nn.Module):
    def __init__(self) -> None:
        super(IdLoss, self).__init__()
        self.mseloss = nn.L1Loss()

    def forward(self, Rec):
        loss = self.mseloss(Rec[0], Rec[1])
        return loss


from utils import pts_1k_to_145_mouth, pts_1k_to_145_eye, pts_1k_to_145_pupil, pts_1k_to_145_others

class LandmarkNet(nn.Module):
    def __init__(self, ckp_path):
        super().__init__()
        self.face_alignment_net = torch.jit.load(ckp_path, map_location='cpu').cuda().eval()
        for param in self.face_alignment_net.parameters():
            param.requires_grad = False

    def forward(self, input):
        bs = input.shape[0]
        landmarks = self.face_alignment_net(input)
        landmarks = landmarks.view(bs, 1000, 2)
        
        landmarks_dict = {}
        landmarks_dict["mouth"] = pts_1k_to_145_mouth(landmarks)
        landmarks_dict["eye"] = pts_1k_to_145_eye(landmarks)
        landmarks_dict["pupil"] = pts_1k_to_145_pupil(landmarks)
        landmarks_dict["others"] = pts_1k_to_145_others(landmarks)
        
        return landmarks_dict
    
class LandmarkLoss(nn.Module):
    def __init__(self, weight={"mouth": 3.0, "eye": 3.0, "pupil": 3.0, "others": 1.0}, ckp_path="./weights/face_alignment_model.pt"):
        super().__init__()
        self.landmarknet = LandmarkNet(ckp_path)
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input, target):
        lm_input = self.landmarknet(input)
        lm_target = self.landmarknet(target)
        
        loss = 0
        for part, weight in self.weight.items():
            loss += weight * self.criterion(lm_input[part], lm_target[part].detach())
        
        return loss


from arcface import iresnet50
from facedet.detector import RetinaFaceDetector
from torchvision import transforms
class ArcfaceLoss(nn.Module):
    # Download the face recognition model arcface from insightface Go to Baidu Drive -> arcface_torch -> glint360k_cosface_r50_fp16_0.1 -> backbone.pth
    def __init__(self, ckp_path="./weights/backbone.pth", if_align=False):
        super().__init__()
        self.arcface = iresnet50().cuda()
        self.arcface.eval()
        self.arcface.load_state_dict(torch.load(ckp_path))
        self.Normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        if if_align:
            self.face_detector = RetinaFaceDetector()
            self.imgdim = 256

    def check_bounding_boxes(self, bounding_boxes):
        filter_boxes = []
        for ix, box in enumerate(bounding_boxes):
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            max_box_len = max(box_w, box_h)
            if box[4] > 0.5 and max_box_len > 100:
                filter_boxes.append([box[0], box[1], box[2], box[3], box[4], box_w * box_h])
        filter_boxes.sort(key=lambda x: x[5], reverse=True)
        face_ratio = 0.1
        filter_boxes = list(filter(lambda x: x[5] > face_ratio*filter_boxes[0][5], filter_boxes))
        return filter_boxes
    
    def face_align(self, img):
        batch_bounding_boxes = self.face_detector.forward(img, min_face_size=120)
        
        theta_list = []
        for idx, bounding_boxes in enumerate(batch_bounding_boxes):
            d = None
            if len(bounding_boxes) == 0:
                d = [0., 0., 256., 256.]
                scale=1
            # if idx == 2:
            #     d = [0., 0., 256., 256.]
            #     scale=1
            bounding_boxes = self.check_bounding_boxes(bounding_boxes)
            if len(bounding_boxes) == 0:
                d = [0., 0., 256., 256.]
                scale=1
            if d is None:
                d = bounding_boxes[0]
                scale = self.imgdim * 0.6 / min(d[2] - d[0], d[3] - d[1])
            src_center = np.array([d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0]).astype(np.float32)
            rotate_degree = 0
            # scale = 1
            dst_center = np.array([0.5, 0.5]) * self.imgdim
            offset = dst_center - src_center
            M = cv2.getRotationMatrix2D((src_center[0], src_center[1]), rotate_degree, 1/scale)
            M[:, 2] += offset
            M[:, 2] = M[:, 2] / self.imgdim
            theta_list.append(torch.from_numpy(np.array(M).astype(np.float32)).cuda().unsqueeze(0))

        theta = torch.cat(theta_list)
        grid = F.affine_grid(theta, [img.shape[0], 3, self.imgdim, self.imgdim], align_corners=True)  # 得到grid 用于grid sample
        warp_img = F.grid_sample(img, grid, align_corners=True)

        return warp_img

    def align_forward(self, generated_d, n, s):
        fake = self.Normalize(self.face_align(torch.flip(generated_d, dims=[1])))
        neutral = self.Normalize(self.face_align(torch.flip(n, dims=[1])))
        real = self.Normalize(self.face_align(torch.flip(s, dims=[1])))

        fake_z_id = self.arcface(F.interpolate(torch.flip(fake, dims=[1]), [143, 143], mode="nearest")[..., 15:127, 15:127])
        neutral_z_id = self.arcface(F.interpolate(torch.flip(neutral, dims=[1]), [143, 143], mode="nearest")[..., 15:127, 15:127])

        with torch.no_grad():
            z_id = self.arcface(F.interpolate(torch.flip(real, dims=[1]), [143, 143], mode="nearest")[..., 15:127, 15:127]).detach()
        L_sim = (1 - torch.cosine_similarity(z_id, fake_z_id)).mean() + 2*(1 - torch.cosine_similarity(z_id, neutral_z_id)).mean()
        return L_sim

    def forward(self, generated_d, n, s):
        fake = self.Normalize(generated_d)
        neutral = self.Normalize(n)
        real = self.Normalize(s)

        fake_z_id = self.arcface(F.interpolate(fake, [143, 143], mode="nearest")[..., 15:127, 15:127])
        neutral_z_id = self.arcface(F.interpolate(neutral, [143, 143], mode="nearest")[..., 15:127, 15:127])

        with torch.no_grad():
            z_id = self.arcface(F.interpolate(real, [143, 143], mode="nearest")[..., 15:127, 15:127])
        L_sim = (1 - torch.cosine_similarity(z_id, fake_z_id)).mean() + 2*(1 - torch.cosine_similarity(z_id, neutral_z_id)).mean()
    
        return L_sim

if __name__ == "__main__":
    from skimage import io, img_as_float32
    img1 = np.array(img_as_float32(io.imread('/home/luh/lh_8T/datasets/vox1/face-video-preprocessing/vox-png/test/id10306#ST2JZXcyYkE#00001.txt#000.mp4/0000000.png'))[:, :, :3]).transpose((2, 0, 1))
    img2 = np.array(img_as_float32(io.imread('/home/luh/lh_8T/datasets/vox1/face-video-preprocessing/vox-png/test/id10306#ST2JZXcyYkE#00001.txt#000.mp4/0000111.png'))[:, :, :3]).transpose((2, 0, 1))
    img3 = np.array(img_as_float32(io.imread('/home/luh/lh_8T/datasets/vox1/face-video-preprocessing/vox-png/test/id10306#ST2JZXcyYkE#00001.txt#000.mp4/0000058.png'))[:, :, :3]).transpose((2, 0, 1))

    imgs1 = torch.cat([torch.from_numpy(img1).cuda().unsqueeze(0), torch.from_numpy(img2).cuda().unsqueeze(0), torch.from_numpy(img3).cuda().unsqueeze(0)])
    imgs2 = torch.cat([torch.from_numpy(img2).cuda().unsqueeze(0), torch.from_numpy(img3).cuda().unsqueeze(0), torch.from_numpy(img1).cuda().unsqueeze(0)])
    imgs3 = torch.cat([torch.from_numpy(img3).cuda().unsqueeze(0), torch.from_numpy(img1).cuda().unsqueeze(0), torch.from_numpy(img2).cuda().unsqueeze(0)])

    A = ArcfaceLoss()
    l = A(imgs1, imgs2, imgs3)
    print(l)


