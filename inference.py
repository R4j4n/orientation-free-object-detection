import cv2
import numpy as np

from yolov5_obb.bare_inference import get_poly_class


class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


hp = Hyperparameters(
    device="0",
    weights="",
    imgsz=[840, 840],
    iou_thres=0.2,  # NMS IOU threshold
    conf_thres=0.6,
    max_det=1000,
    img_pth="",
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
for k, v in resp.items():
    ploy = v["polygons"]
    pts = np.array(
        [
            [ploy[0], ploy[1]],
            [ploy[2], ploy[3]],
            [ploy[4], ploy[5]],
            [ploy[6], ploy[7]],
        ],
        np.int32,
    )

    pts = pts.reshape((-1, 1, 2))

    isClosed = True
    color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.polylines() method
    # Draw a Blue polygon with
    # thickness of 1 px
    image = cv2.polylines(img, [pts], isClosed, color, thickness)

cv2.imwrite("result.png", img)
