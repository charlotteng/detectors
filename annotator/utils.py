from canny import CannyDetector
from midas import MidasDetector
from hed import HEDdetector, nms
from mlsd import MLSDdetector
from openpose import OpenposeDetector
from uniformer import UniformerDetector
from util import resize_image, HWC3

import numpy as np
import cv2
import os


annotator_ckpts_path = os.path.join(os.path.dirname(__file__), 'ckpts')


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def create_detector(model_name): 
    
    detectors = {
        'canny': CannyDetector(),
        'mlsd': MLSDdetector(),
        'hed': HEDdetector(),
        'scribble': 'no detector required',
        'openpose': OpenposeDetector(),
        'seg': UniformerDetector(),
        'depth': MidasDetector(),
        'normal': MidasDetector()
    }

    detector = detectors[model_name]

    return detector


def controlnet_map(model_name, detector, input_image, opts):

  #canny
  if model_name == "canny":
    
    img = resize_image(HWC3(input_image), opts.image_resolution)
    detected_map = detector(img, opts.low_threshold, opts.high_threshold)
    detected_map = HWC3(detected_map)

  #depth / hed
  if model_name == "depth" or model_name == "hed":
    
    input_image = HWC3(input_image)
    detected_map, _ = detector(resize_image(input_image, opts.detect_resolution))
    detected_map = HWC3(detected_map)

    img = resize_image(input_image, opts.image_resolution)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map,
                              (W, H),
                              interpolation=cv2.INTER_NEAREST)

  #hough
  if model_name == "mlsd":
    
    input_image = HWC3(input_image)
    detected_map = detector(
        resize_image(input_image, opts.detect_resolution),
        opts.value_threshold, opts.distance_threshold)
    detected_map = HWC3(detected_map)

    img = resize_image(input_image, opts.image_resolution)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map,
                              (W, H),
                              interpolation=cv2.INTER_NEAREST)
  
  #normal  
  if model_name == "normal":
    
    input_image = HWC3(input_image)
    _, detected_map = detector(
        resize_image(input_image, opts.detect_resolution),
        bg_th=opts.bg_threshold)
    detected_map = HWC3(detected_map)
    
    img = resize_image(input_image, opts.image_resolution)
    H, W, C = img.shape
    
    detected_map = cv2.resize(detected_map,
                              (W, H),
                              interpolation=cv2.INTER_LINEAR)
  
  #pose
  if model_name == "openpose":
    input_image = HWC3(input_image)
    detected_map, _ = detector(resize_image(input_image, opts.detect_resolution))
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, opts.image_resolution)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map,
                              (W, H),
                              interpolation=cv2.INTER_NEAREST)

  #scribble
  if model_name == "scribble":
    
    img = resize_image(HWC3(input_image), opts.image_resolution)
    H, W, C = img.shape

    detected_map = np.zeros_like(img, dtype=np.uint8)
    detected_map[np.min(img, axis=2) < 127] = 255

  #seg
  if model_name == "seg":
    
    input_image = HWC3(input_image)
    detected_map = detector(resize_image(input_image, opts.detect_resolution))
    img = resize_image(input_image, opts.image_resolution)
    
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map,
                              (W, H),
                              interpolation=cv2.INTER_NEAREST)
  
  return detected_map

def dynamic_canny_threshold(input_image):
    
    gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    high_threshold, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_threshold = 0.5* high_threshold

    return low_threshold, high_threshold
