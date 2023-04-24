import math
import numpy as np
from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square, _preprocess
import torch
import cv2
from .nms.py_cpu_nms import py_cpu_nms
from . import box_utils_Retina
from .layers.functions.prior_box import PriorBox
from .config import cfg_re50
from .retinaface import RetinaFace
from torch.nn import functional as F

class RetinaFaceDetector(object):
    def __init__(self, gpu_id=None):
        self.cfg = cfg_re50
        # self.device = torch.device('cuda:{}'.format(gpu_id) if gpu_id is not None else 'cpu')
        model = RetinaFace(cfg=self.cfg, phase='test')
        model = self.load_model(model, './weights/Resnet50_Final.pth', True)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.net = model.cuda()
        self.resize = 1
        self.confidence_threshold = 0.02
        self.top_k = 500
        self.nms_threshold = 0.4
        self.keep_top_k = 100
        self.im_dim = 256
        priorbox = PriorBox(self.cfg, image_size=(self.im_dim, self.im_dim))
        with torch.no_grad():
            priors = priorbox.forward()
            self.priors = priors.cuda()

    def remove_prefix(self, state_dict, prefix):
        # print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        # unused_pretrained_keys = ckpt_keys - model_keys
        # missing_keys = model_keys - ckpt_keys
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def load_model(self, model, pretrained_path, load_to_cpu):
        # print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def forward(self, img_tensor, min_face_size=120):
        img_scale = self.im_dim / max(img_tensor.shape[2], img_tensor.shape[3])
        # img = cv2.resize(img_raw, (0, 0), fx=img_scale, fy=img_scale)
        img = F.interpolate(img_tensor, scale_factor=img_scale, mode='bicubic', align_corners=True)
        img = torch.clamp(img*255, 0, 255)
        batch, channel, im_height, im_width = img.shape
        scale = torch.Tensor([im_width, im_height, im_width, im_height])
        img -= torch.Tensor([104., 117., 123.])[:, None, None].cuda()
        # img = img.transpose(2, 0, 1)
        # img = torch.from_numpy(img).unsqueeze(0)
        # img = img.to(self.device)
        scale = scale.cuda()
        loc, conf, landms = self.net(img)  # forward pass
        prior_data = self.priors.data
        batch_dets = []
        for idx in range(batch):
            boxes = box_utils_Retina.decode(loc.data[idx], prior_data, self.cfg['variance'])
            boxes = boxes * scale / self.resize

            boxes = boxes.cpu().numpy()
            scores = conf[idx].data.cpu().numpy()[:, 1]
            # landms = box_utils_Retina.decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
            # scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
            #                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
            #                        img.shape[3], img.shape[2]])
            # scale1 = scale1.to(self.device)
            # landms = landms * scale1 / self.resize
            # landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > self.confidence_threshold)[0]
            boxes = boxes[inds]
            # landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:self.top_k]
            boxes = boxes[order]
            # landms = landms[order]
            # scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.nms_threshold, min_face_size = min_face_size)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            # landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:self.keep_top_k, :]
            # landms = landms[:self.keep_top_k, :]

            dets[:, :4] = dets[:, :4] / img_scale
            # landms /= img_scale
            # return dets, landms
            batch_dets.append(dets)
        return batch_dets


def run_first_stage(image, net, scale, threshold, gpu_id=0):
    """ 
        Run P-Net, generate bounding boxes, and do NMS.
    """
    device = torch.device('cuda:{}'.format(gpu_id) if gpu_id is not None else 'cpu')
    (height, width, _) = image.shape
    sw, sh = math.ceil(width * scale), math.ceil(height * scale)
    img = cv2.resize(image, (sw, sh))
    # img = image.resize((sw, sh), Image.BILINEAR)
    img = np.asarray(img, 'float32')
    img = torch.from_numpy(_preprocess(img))
    img = img.to(device)

    output = net(img)
    probs = output[1].to('cpu').data.numpy()[0, 1, :, :]
    offsets = output[0].to('cpu').data.numpy()

    boxes = _generate_bboxes(probs, offsets, scale, threshold)
    if len(boxes) == 0:
        return None

    keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
    return boxes[keep]


def _generate_bboxes(probs, offsets, scale, threshold):
    """
       Generate bounding boxes at places where there is probably a face.
    """
    stride = 2
    cell_size = 12

    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return np.array([])

    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    # P-Net is applied to scaled images, so we need to rescale bounding boxes back
    bounding_boxes = np.vstack([
        np.round((stride * inds[1] + 1.0) / scale),
        np.round((stride * inds[0] + 1.0) / scale),
        np.round((stride * inds[1] + 1.0 + cell_size) / scale),
        np.round((stride * inds[0] + 1.0 + cell_size) / scale),
        score, offsets
    ])

    return bounding_boxes.T
