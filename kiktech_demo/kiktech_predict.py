import os

from pycocotools.coco import COCO
from demo.predictor import COCODemo
from maskrcnn_benchmark.config import cfg
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import cv2

# set image dpi
plt.rcParams['figure.dpi'] = 300

# define the address of the image files
dataset_dir = './datasets/kiktech/skyeye_data'
annotations_dir = '{}/annotations'.format(dataset_dir)
test_images_dir = '{}/test'.format(dataset_dir)
test_ann_file = '{}/kiktech_test.json'.format(annotations_dir)
coco = COCO(test_ann_file)

config_file = './configs/e2e_mask_rcnn_R_50_FPN_1x_copy.yaml'
cfg.merge_from_file(config_file)
coco_demo = COCODemo(
    cfg,
    # TODO: add confidence threshold
)


def compute_label_color(labels):
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    labels = np.array(labels)
    colors = labels[:, None] * palette
    colors = (colors % 255).astype('uint8')
    return colors


def evaluation_in_coco(f_image_path):
    I = cv2.imread(f_image_path)
    plt.axis('off')
    plt.imshow(I)
    # plt.show()
    annIds = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(annIds)
    labels = [anns[k]['category_id'] for k in range(len(anns))]

    for j in range(len(labels)):
        if labels[j] == -1:
            labels[j] = 0

    colors = compute_label_color(labels).tolist()
    for n in range(len(anns)):
        x, y, w, h = anns[n]['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        if anns[n]['category_id'] != -1:
            text = coco.loadCats(anns[n]['category_id'])[0]['name']
        else:
            text = 'ignore'
        cv2.putText(I, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.rectangle(I, (x, y), (x + w, y + h), tuple(colors[n]), 1)
    coco.showAnns(anns)
    plt.imshow(I)
    plt.show()


def evaluation_in_model(f_image_path):
    I = cv2.imread(f_image_path)
    composite = coco_demo.run_on_opencv_image(I)
    cv2.imshow("predict detections", composite)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    categories = coco.loadCats(coco.getCatIds())
    print(categories)
    # for n in range(len(categories)):
    #     print(categories[n]['name'])
    catIds = coco.getCatIds(catNms=['mask_person_top'])
    # print(len(coco.loadCats(coco.getCatIds())))
    imgIds = coco.getImgIds()
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    image_path = test_images_dir + '/{}'.format(img['file_name'])

    evaluation_in_coco(image_path)
    evaluation_in_model(image_path)
