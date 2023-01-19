from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

import torch


PATH_IMAGE = Path("datasets", "images", "validation", "01d5f28e-fyziologicky_nalez_frame_254.jpg")
PATH_LABEL = Path("datasets", "labels", "validation", "01d5f28e-fyziologicky_nalez_frame_254.txt")

img = mpimg.imread(PATH_IMAGE)
with open(PATH_LABEL, "r") as f:
    _, rect_center_x, rect_center_y, rect_width_rel, rect_height_rel = f.read().split(" ")

height, width, _ = img.shape
rect_x = (float(rect_center_x) - float(rect_width_rel) / 2) * width
rect_y = (float(rect_center_y) - float(rect_height_rel) / 2) * height
rect_height = float(rect_height_rel) * height
rect_width = float(rect_width_rel) * width

rect_original = Rectangle((rect_x, rect_y), rect_width, rect_height, fill=False, edgecolor="midnightblue", linewidth=1.5)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='trained_models/best.pt')
x_min, y_min, x_max, y_max, _, _, _ = model(img).pandas().xyxy[0].iloc[0, :]

rect_yolo = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor="darkgoldenrod", linewidth=1.5)

fig, ax = plt.subplots(1, 1, figsize=(5, 3.75))

ax.imshow(img)
ax.set_xticks([])
ax.set_yticks([])

ax.add_patch(rect_original)
ax.add_patch(rect_yolo)
# ax.set_title("Comparison of Label and Inference", fontsize=25, pad=20)
ax.legend([rect_original, rect_yolo], ["Original Bounding Box", "YOLO Object Detection"])
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0, hspace=0)
plt.tight_layout()
fig.savefig("comparison.png", bbox_inches='tight', transparent="True", pad_inches=0)
print("Original bounding box:\nxmin: {:.2f}, ymin: {:.2f}, xmax: {:.2f}, ymax: {:.2f}"
      .format(rect_x, rect_y, rect_x + rect_width, rect_y + rect_height))
print("YOLO bounding box:\nxmin: {:.2f}, ymin: {:.2f}, xmax: {:.2f}, ymax: {:.2f}"
      .format(x_min, y_min, x_max, y_max))
intersection = max(0, min(rect_x + rect_width, x_max) - max(rect_x, x_min)) * \
    max(0, min(rect_y + rect_height, y_max) - max(rect_y, y_min))
union = rect_width * rect_height + (x_max - x_min) * (y_max - y_min) - intersection
iou = intersection / union
print("IOU: {:.2%}".format(iou))



