from annotator.canny import CannyDetector
from annotator.midas import MidasDetector
from annotator.hed import HEDdetector, nms
from annotator.mlsd import MLSDdetector
from annotator.openpose import OpenposeDetector
#from annotator.uniformer import UniformerDetector
from annotator.util import resize_image, HWC3

import cv2

def create_detector(model_name): 
    
    detectors = {
        'canny': CannyDetector(),
        'mlsd': MLSDdetector(),
        'hed': HEDdetector(),
        'scribble': 'no detector required',
        'openpose': OpenposeDetector(),
        'depth': MidasDetector(),
        'normal': MidasDetector(),
        #'seg': UniformerDetector(),
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
  if model_name == "depth":
    
    input_image = HWC3(input_image)
    detected_map, _ = detector(resize_image(input_image, opts.detect_resolution))
    detected_map = HWC3(detected_map)

    img = resize_image(input_image, opts.image_resolution)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map,
                              (W, H),
                              interpolation=cv2.INTER_NEAREST)
  #hed
  if model_name == "hed":
    
    input_image = HWC3(input_image)
    detected_map = detector(resize_image(input_image, opts.detect_resolution))
    detected_map = HWC3(detected_map)

    img = resize_image(input_image, opts.image_resolution)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map,
                              (W, H),
                              interpolation=cv2.INTER_LINEAR)

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
