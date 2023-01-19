import pandas as pd
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from random import shuffle
import cv2

import torch

LABELED_IMAGES_FOLDER = Path(".", "datasets", "images", "validation")
LABELS_FOLDER = Path(".", "datasets", "labels", "validation")
UNLABELED_IMAGES_FOLDER = Path(".", "frames", "bad")

# List of labeled and unlabeled images
labeled_file_names = [x for x in LABELED_IMAGES_FOLDER.iterdir()]
unlabeled_file_names = [x for x in UNLABELED_IMAGES_FOLDER.glob('**/*') if x.is_file()]
shuffle(unlabeled_file_names)

# Coordinates of labeled and unlabeled images
image_width = 960
image_height = 720
labeled_file_x_coords = []
labeled_file_y_coords = []

for file in labeled_file_names:
    with open(LABELS_FOLDER.joinpath(f"{file.name.split('.')[0]}.txt")) as f:
        _, rect_center_x, rect_center_y, rect_width_rel, rect_height_rel = f.read().split(" ")

    labeled_file_x_coords.append(((float(rect_center_x) - float(rect_width_rel) / 2) * image_width,
                                  (float(rect_center_x) + float(rect_width_rel) / 2) * image_width))
    labeled_file_y_coords.append(((float(rect_center_y) - float(rect_height_rel) / 2) * image_height,
                                  (float(rect_center_y) + float(rect_height_rel) / 2) * image_height))

unlabeled_file_x_coords = [(0, 0) for x in unlabeled_file_names]
unlabeled_file_y_coords = [(0, 0) for x in unlabeled_file_names]

model = torch.hub.load('ultralytics/yolov5', 'custom', path='trained_models/best.pt')
# model.iou = 0.5
# model.conf = 0.00
yolo_file_name = []
yolo_x_coords = []
yolo_y_coords = []
yolo_center = []
yolo_modded = []
# for labeled_img, unlabeled_img in zip(labeled_file_names, unlabeled_file_names[:len(labeled_file_names)]):
for labeled_img, unlabeled_img in zip(labeled_file_names[:5], unlabeled_file_names[:5]):
    regular_img = cv2.imread(labeled_img)
    regular_img = cv2.cvtColor(regular_img, cv2.COLOR_BGR2RGB)
    irregular_img = cv2.imread(unlabeled_img)
    irregular_img = cv2.cvtColor(irregular_img, cv2.COLOR_BGR2RGB)
    modded_img = cv2.hconcat([regular_img, irregular_img])
    for j, img in enumerate([regular_img, modded_img]):
        result = model(img)
        result.show()
        for i in range(result.pandas().xyxy[0].shape[0]):
            yolo_file_name.append(labeled_img.name)
            yolo_modded.append([unlabeled_img.name if j == 1 else "no"])
            yolo_x_coords.append((result.pandas().xyxy[0].xmin.values[i], result.pandas().xyxy[0].xmax.values[i]))
            yolo_y_coords.append((result.pandas().xyxy[0].ymin.values[i], result.pandas().xyxy[0].ymax.values[i]))
            yolo_center.append(((yolo_x_coords[-1][0] + yolo_x_coords[-1][1]) / 2,
                               (yolo_y_coords[-1][0] + yolo_y_coords[-1][1]) / 2))

df_yolo = pd.DataFrame(list(zip(yolo_file_name, yolo_modded, yolo_x_coords, yolo_y_coords, yolo_center)),
                       columns=["name", "modded", "x", "y", "center"])
df_yolo.to_excel("validation_data.xlsx")

# Intersection in one direction - min of max coordinates minus maximum of min coordinates => max of 0 and final difference


# index = 150
# rect_x = labeled_file_x_coords[index][0]
# rect_y = labeled_file_y_coords[index][0]
# rect_width = labeled_file_x_coords[index][1] - labeled_file_x_coords[index][0]
# rect_height = labeled_file_y_coords[index][1] - labeled_file_y_coords[index][0]
#
# rect_original = Rectangle((rect_x, rect_y), rect_width, rect_height, fill=False, edgecolor="midnightblue", linewidth=1.5)
#
# img = mpimg.imread(LABELED_IMAGES_FOLDER.joinpath(f"{labeled_file_names[index]}.jpg"))
# fig, ax = plt.subplots(1, 1, figsize=(5, 3.75))
#
# ax.imshow(img)
# ax.set_xticks([])
# ax.set_yticks([])
#
# ax.add_patch(rect_original)
# plt.show()