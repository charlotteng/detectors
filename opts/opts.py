
class OptsCanny:

  def __init__(self, image_resolution=512, low_threshold=100, high_threshold=200):
    
    self.image_resolution = image_resolution #min=256, max=768
    self.low_threshold = low_threshold # 1 to 255
    self.high_threshold = high_threshold # 1 to 255

class OptsDepth:

  def __init__(self, image_resolution=512, detect_resolution=384):
    
    self.image_resolution = image_resolution #min=256, max=768
    self.detect_resolution = detect_resolution #128 to 1024

class OptsHed:

  def __init__(self, image_resolution=512, detect_resolution=384):
    
    self.image_resolution = image_resolution #min=256, max=768
    self.detect_resolution = detect_resolution #128 to 1024

class OptsMLSD:

  def __init__(self, image_resolution=512, detect_resolution=384,
               value_threshold=0.1, distance_threshold=0.12):
    
    self.image_resolution = image_resolution #min=256, max=768
    self.detect_resolution = detect_resolution #128 to 1024
    self.value_threshold = value_threshold #0.01 to 2.0
    self.distance_threshold = distance_threshold #0.01 to 20

class OptsSeg:

  def __init__(self, image_resolution=512, detect_resolution=384):
    
    self.image_resolution = image_resolution #min=256, max=768
    self.detect_resolution = detect_resolution #128 to 1024

class OptsNormal:

  def __init__(self, image_resolution=512, detect_resolution=384,
               bg_threshold=0.4):
  
    self.image_resolution = image_resolution #min=256, max=768
    self.detect_resolution = detect_resolution #128 to 1024
    self.bg_threshold = bg_threshold #0.0 to 1.0

class OptsOpenPose:

  def __init__(self, image_resolution=512, detect_resolution=384):
    
    self.image_resolution = image_resolution #min=256, max=768
    self.detect_resolution = detect_resolution #128 to 1024

class OptsScribble:

  def __init__(self, image_resolution=512):
    
    self.image_resolution = image_resolution #min=256, max=768
