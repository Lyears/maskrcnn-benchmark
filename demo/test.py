from maskrcnn_benchmark.config import cfg
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pylab as plt
from predictor import COCODemo

import cv2

config_file = '../configs/e2e_mask_rcnn_R_50_FPN_1x_copy.yaml'

cfg.merge_from_file(config_file)
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    confidence_threshold=0.7,
)
ann_val_file = '../datasets/coco/annotations/instances_val2014.json'
coco = COCO(ann_val_file)


def predict(f_image_):
    composite = coco_demo.run_on_opencv_image(f_image_)
    cv2.imshow("predict detections", composite)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def coco_anns(image_path):
#     f_image = cv2.imread(image_path)
#     plt.axis('off')
#     plt.imshow(f_image)
#     plt.show()


if __name__ == '__main__':
    # image_file_name = 'COCO_val2014_000000000294.jpg'
    # image_path = '../datasets/coco/val2014/COCO_val2014_000000000294.jpg'
    catIds = coco.getCatIds(catNms=['person', 'dog'])
    imgIds = coco.getImgIds(catIds=catIds)

    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    image_path = '../datasets/coco/val2014/{}'.format(img['file_name'])

    image = cv2.imread(image_path)
    f_image = cv2.imread(image_path)
    # coco_anns(image_path)
    plt.axis('off')
    plt.imshow(image)

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    for n in range(len(anns)):
        x, y, w, h = anns[n]['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        text = coco.loadCats(anns[n]['category_id'])[0]['name']
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    plt.imshow(image)
    plt.show()

    predict(f_image)
