from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import cv2

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

cfg.merge_from_file(config_file)
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

image = cv2.imread('timg.jpeg')

while True:
    composite = coco_demo.run_on_opencv_image(image)
    cv2.imshow("COCO detections", composite)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
