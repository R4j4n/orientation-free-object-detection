import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_obb, print_args, scale_coords, scale_polys, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly
from PIL import Image

import numpy as np

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad image while meeting stride-multiple constraints
    Returns:
        im (array): (height, width, 3)
        ratio (array): [w_ratio, h_ratio]
        (dw, dh) (array): [w_padding h_padding]
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int): # [h_rect, w_rect]
        new_shape = (new_shape, new_shape)


    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # wh ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) # w h 
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0]) # [w h]
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # [w_ratio, h_ratio]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)



def get_poly_class(hp):
    device = select_device(hp.device)
    model = DetectMultiBackend(hp.weights, device=device, dnn=hp.dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(hp.imgsz, s=stride)  

    hp.half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA    
    img0 = cv2.imread(hp.img_pth)
    
    # Padded resize
    img = letterbox(img0, imgsz, stride=stride, auto=pt)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    
    im = torch.from_numpy(img).to(device)
    im = im.half() if hp.half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None] 
    
    model.warmup(imgsz=(1, 3, *imgsz), half=hp.half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    
    
    pred = model(im, augment=hp.augment, visualize=False)
    
    pred = non_max_suppression_obb(pred, hp.conf_thres, hp.iou_thres, hp.classes, hp.agnostic_nms, multi_label=True, max_det=hp.max_det)

    mapping = {}
    for i, det in enumerate(pred):  # per image
        pred_poly = rbox2poly(det[:, :5]) 

        
        if len(det):
            # Rescale polys from img_size to im0 size
            # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            pred_poly = scale_polys(im.shape[2:], pred_poly, img0.shape)
            det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])
            cnt = 0
            
            for *poly, conf, cls in reversed(det):
                x_ = [int(x.detach().cpu().numpy()) for x in poly]                
                mapping[cnt] = {
                    "polygons" : x_,  
                    "class"    : int(cls.detach().cpu().numpy()),
                    "conf"     : conf.detach().cpu().numpy().item()
                }
                cnt += 1 
                
    
    return mapping

class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
if __name__ == "__main__":
    
    hp = Hyperparameters(
        device = '0',
        weights = "/home/ml/rajan/YOLO/results/train/exp/weights/best.pt",
        imgsz=[840,840],
        iou_thres=0.1,  # NMS IOU threshold
        conf_thres=0.1,
        max_det=1000,
        img_pth = "/home/ml/rajan/YOLO/data/dota_rajan/P0047.png",
        half=False,  # use FP16 half-precision inference
        dnn=False,
        augment=False,
        classes=None,
        agnostic_nms=False,

    )

    resp = get_poly_class(hp)
    print(resp)
   
    
    img = cv2.imread(hp.img_pth)
    # plot the poly to validate
    for k , v in resp.items():

        ploy = v["polygons"]
        pts = np.array([[ploy[0], ploy[1]], [ploy[2], ploy[3]],
                        [ploy[4], ploy[5]], [ploy[6], ploy[7]]],
                    np.int32)
        
        pts = pts.reshape((-1, 1, 2))
        
        isClosed = True
        color = (0, 0, 255)
 
        # Line thickness of 2 px
        thickness = 2
        
        # Using cv2.polylines() method
        # Draw a Blue polygon with
        # thickness of 1 px
        image = cv2.polylines(img, [pts],
                            isClosed, color, thickness)
        
    cv2.imwrite("Img.png",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()