"""Config for the ChimpDetectorPredictor Class
"""

IM_RESIZE_HEIGHT = 1352
IM_RESIZE_WIDTH = 1688
CLAHE_GRID_SIZE = (12, 12)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_HIDDEN_LAYER_SIZE = 256
EDGE_KERNEL = [[-1,-1,-1],
               [-1, 8,-1],
               [-1,-1,-1]]