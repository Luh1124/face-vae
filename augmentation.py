"""
Code from https://github.com/hassony2/torch_videovision
"""
import numbers

import random
import numpy as np
import PIL
from PIL import ImageFilter, Image

import torchvision
import torchvision.transforms as transforms

import cv2
from numpy import pad

import warnings
from skimage.transform import resize, rotate

from skimage import img_as_ubyte, img_as_float, img_as_float32

def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
            ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def pad_clip(clip, h, w):
    im_h, im_w = clip[0].shape[:2]
    pad_h = (0, 0) if h < im_h else ((h - im_h) // 2, (h - im_h + 1) // 2)
    pad_w = (0, 0) if w < im_w else ((w - im_w) // 2, (w - im_w + 1) // 2)

    return pad(clip, ((0, 0), pad_h, pad_w, (0, 0)), mode='edge')


def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]

        scaled = [
            resize(img, size, order=1 if interpolation == 'bilinear' else 0, preserve_range=True,
                   mode='constant', anti_aliasing=True) for img in clip
            ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


class RandomResize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = resize_clip(
            clip, new_size, interpolation=self.interpolation)

        return resized

class RandomCrop(object):
    """Extract random crop at the same location for a list of videos
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of videos to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of videos
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        clip = pad_clip(clip, h, w)
        im_h, im_w = clip.shape[1:3]
        x1 = 0 if h == im_h else random.randint(0, im_w - w)
        y1 = 0 if w == im_w else random.randint(0, im_h - h)
        cropped = crop_clip(clip, y1, x1, h, w)

        return cropped

class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of videos to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of videos
        """
        out = clip
        for i in range(len(clip)):
            angle = random.uniform(self.degrees[0], self.degrees[1])
            h, w, c = clip[i].shape
            warped = cv2.warpAffine(src = clip[i],  # 原图像
                                    # 仿射变换的旋转矩阵参数
                                M = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=angle, scale=1),  
                                dsize = (w, h),   # 原尺寸大小（注意这里的顺序是相反的）
                                borderMode=cv2.BORDER_REPLICATE)  # 填充值
            out[i] = warped
        return out

class RandomFlip(object):
    def __init__(self, time_flip=False, horizontal_flip=False):
        self.time_flip = time_flip
        self.horizontal_flip = horizontal_flip

    def __call__(self, clip):
        if random.random() < 0.5 and self.time_flip:
            return clip[::-1]
        if random.random() < 0.5 and self.horizontal_flip:
            return [np.fliplr(img) for img in clip]

        return clip


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            brightness, contrast, saturation, hue = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)
            img_transforms = [img_as_ubyte, torchvision.transforms.ToPILImage()] + img_transforms + [np.array, img_as_float]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jittered_clip = []
                for img in clip:
                    jittered_img = img
                    for func in img_transforms:
                        jittered_img = func(jittered_img)
                    jittered_clip.append(jittered_img.astype("float32"))
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all videos
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)

        else:
            raise TypeError("Expected numpy.ndarray or PIL.Image" + "but got list of {0}".format(type(clip[0])))
        return jittered_clip


class RandomPerspective(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, pers_num, enlarge_num):
        self.pers_num = pers_num
        self.enlarge_num = enlarge_num

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of videos to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of videos
        """
        out = clip
        for i in range(len(clip)):
            self.pers_size = np.random.randint(20, self.pers_num) * pow(-1, np.random.randint(2))
            self.enlarge_size = np.random.randint(20, self.enlarge_num) * pow(-1, np.random.randint(2))
            h, w, c = clip[i].shape
            crop_size=256
            dst = np.array([
                [-self.enlarge_size, -self.enlarge_size],
                [-self.enlarge_size + self.pers_size, w + self.enlarge_size],
                [h + self.enlarge_size, -self.enlarge_size],
                [h + self.enlarge_size - self.pers_size, w + self.enlarge_size],], dtype=np.float32)
            src = np.array([[-self.enlarge_size, -self.enlarge_size], [-self.enlarge_size, w + self.enlarge_size],
                        [h + self.enlarge_size, -self.enlarge_size], [h + self.enlarge_size, w + self.enlarge_size]]).astype(np.float32())
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(clip[i], M, (crop_size, crop_size), borderMode=cv2.BORDER_REPLICATE)
            out[i] = warped

        return out


class RandomScale(object):

    def __init__(self, ratio = [0.7, 1.2]):
        self.ratio = ratio

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of videos to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of videos
        """
        out = clip
        for i in range(len(clip)):
            scale_factor = np.random.uniform(self.ratio[0], self.ratio[1])
            h, w, c = clip[i].shape
            warped = cv2.warpAffine(src = clip[i],  # 原图像
                                    # 仿射变换的旋转矩阵参数
                                M = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=0, scale=scale_factor),  
                                dsize = (w, h),   # 原尺寸大小（注意这里的顺序是相反的）
                                borderMode=cv2.BORDER_REPLICATE)  # 填充值
            out[i] = warped
        return out


class RandomTranslate(object):
    def __init__(self, x_ratio=1/16., y_ratio=1/16.):
        self.x_ratio = x_ratio
        self.y_ratio = y_ratio

    def __call__(self, clip):
        h, w, c = clip[0].shape
        x_delta, y_delta = int(self.x_ratio*w), int(self.y_ratio*h)
        out = clip
        for i in range(len(clip)):
            tx = np.random.randint(-1*x_delta, x_delta)
            ty = np.random.randint(-1*y_delta, y_delta)
            M = np.array([
                [1, 0, tx],
                [0, 1, ty]
            ], dtype=np.float32)
            warped = cv2.warpAffine(clip[i], M, dsize = (w, h), borderMode=cv2.BORDER_REPLICATE)
            out[i] = warped

        return out

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, clip):
        out = clip
        for i in range(len(clip)):
            self.gaussianblur= transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=self.sigma)  # not strengthened
            ], p=0.5)
            out[i] = Image.fromarray(np.uint8(out[i][:,:,:3]))
            out[i] = self.gaussianblur(out[i])
        return out

class RandomGrayscale(object):
    """RandomGrayscale augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, p=0.5):
        self.RandomGrayscale = transforms.RandomGrayscale(p=p)

    def __call__(self, clip):
        out = clip
        for i in range(len(clip)):
            out[i]= self.RandomGrayscale(out[i])
        return out

class AllAugmentationTransform:
    def __init__(self, resize_param=None, rotation_param=None, perspective_param=None, 
                 flip_param=None, crop_param=None, jitter_param=None, blur_param=None, 
                 gray_param=None, scale_param=None, translate_param=None):

        self.transforms = []

        if flip_param is not None:
            self.transforms.append(RandomFlip(**flip_param))

        if rotation_param is not None:
            self.transforms.append(RandomRotation(**rotation_param))

        if perspective_param is not None:
            self.transforms.append(RandomPerspective(**perspective_param))

        if resize_param is not None:
            self.transforms.append(RandomResize(**resize_param))

        if crop_param is not None:
            self.transforms.append(RandomCrop(**crop_param))

        if jitter_param is not None:
            self.transforms.append(ColorJitter(**jitter_param))
        
        # if blur_param is not None:
        #     self.transforms.append(GaussianBlur(**blur_param))

        # if gray_param is not None:
        #     self.transforms.append(RandomGrayscale(**gray_param))
        
        if scale_param is not None:
            self.transforms.append(RandomScale(**scale_param))

        if translate_param is not None:
            self.transforms.append(RandomTranslate(**translate_param))

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip

if __name__ == "__main__":
    import skimage.io as io
    from skimage import transform
    # import imageio
    x = io.imread("/home/lh/repo/code/lh/1.faceanimation/face-vae/demo_image/R-C.png")
    x = transform.resize(x, (256,256))
    x = img_as_float32(x)
    # x = PIL.Image.fromarray(x)
    # print(len(x))
    print(x.shape)
    # Gblur=RandomPerspective(pers_num=30, enlarge_num=40)
    # img = cv2.imread("./photo/cow.jpg")
    # h, w = x.shape[0], x.shape[1]
    # r_img = cv2.warpAffine(src = x,    # 原图像
    #                         # 仿射变换的旋转矩阵参数
    #                     M = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=0, scale=0.8),  
    #                     dsize = (w, h),   # 原尺寸大小（注意这里的顺序是相反的）
    #                     borderValue = (0, 0, 0))                # 填充值
    # Gblur = RandomRotation(30)
    augmentation_params={
            "rotation_param": {"degrees": 30},
            # "perspective_param":{"pers_num": 30, "enlarge_num": 40},
            # "flip_param": {"horizontal_flip": False, "time_flip": False},
            "scale_param": {"ratio": [0.75, 1.2]},
            "translate_param": {"x_ratio": 1/16, "y_ratio": 1/16}, 
            "jitter_param": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
        }
    
    Gblur = AllAugmentationTransform(**augmentation_params)
    
    y = Gblur([x])[0]
    
    imgpath = "/home/lh/repo/code/lh/1.faceanimation/face-vae/demo_image/R-C-pro.png"
    io.imsave(imgpath, img_as_ubyte(y))