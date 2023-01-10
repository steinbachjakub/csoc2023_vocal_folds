from pathlib import Path
import json
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from numpy.random import default_rng

EXPORT_PATH = Path("export")
IMAGE_PATH = EXPORT_PATH.joinpath("images")
LABEL_PATH = EXPORT_PATH.joinpath("labels")
CLASSES_NAME = EXPORT_PATH.joinpath("classes.txt")

if not EXPORT_PATH.exists():
    EXPORT_PATH.mkdir()
if not IMAGE_PATH.exists():
    IMAGE_PATH.mkdir()
    print(f"Copy images in {IMAGE_PATH.resolve()}.")
if not LABEL_PATH.exists():
    LABEL_PATH.mkdir()

with open("labels.json") as f:
    labels_dict = json.loads(f.read())

# jméno txt souboru = item["annotations"][0][file_upload]
# šířka souboru = item["annotations"][0][result][original_width]
# výška souboru = item["annotations"][0][result][original_height]
# x-ová pozice anotace = item["annotations"][0][result][x]
# y-ová pozice anotace = item["annotations"][0][result][y]
# šířka anotace = item["annotations"][0][result][width]
# výška pozice anotace = item["annotations"][0][result][height]
file_name = []
center_x = []
center_y = []
width = []
height = []

for item in labels_dict:
    if len(item["annotations"][0]["result"]) > 0:
        img_width = item["annotations"][0]["result"][0]["original_width"]
        img_height = item["annotations"][0]["result"][0]["original_height"]
        x_coord = item["annotations"][0]["result"][0]["value"]["x"]
        y_coord = item["annotations"][0]["result"][0]["value"]["y"]
        frame_width = item["annotations"][0]["result"][0]["value"]["width"]
        frame_height = item["annotations"][0]["result"][0]["value"]["height"]

        file_name = item["file_upload"].split(".")[0]
        center_x = 1/100 * (x_coord + frame_width / 2)
        center_y = 1/100 * (y_coord + frame_height / 2)
        width = frame_width / 100
        height = frame_height / 100

        with open(LABEL_PATH.joinpath(f"{file_name}.txt"), "w") as f:
            f.write(f"0 {center_x} {center_y} {width} {height}")

with open(EXPORT_PATH.joinpath("classes.txt"), "w") as f:
    f.write("vocal_cords")
