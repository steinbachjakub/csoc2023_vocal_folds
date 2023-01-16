from pathlib import Path
import json
import random
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from numpy.random import default_rng

# TODO: Rozdělit data na trénovací a validační tak, aby bylo rovnoměrné zastoupení ze všech tří nahrávek
# TODO: Pořešit, co s tím YAML souborem, který deifnuje jednotlivé cesty
# TODO: Zjistit, jak spustit ten posranej YOLO

# Paths
EXPORT_PATH = Path("datasets")
IMAGE_PATH = EXPORT_PATH.joinpath("images")
LABEL_PATH = EXPORT_PATH.joinpath("labels")
CLASSES_NAME = EXPORT_PATH.joinpath("classes.txt")
IMG_TRAINING_PATH = IMAGE_PATH.joinpath("training")
IMG_VALIDATION_PATH = IMAGE_PATH.joinpath("validation")
TXT_TRAINING_PATH = LABEL_PATH.joinpath("training")
TXT_VALIDATION_PATH = LABEL_PATH.joinpath("validation")

if not EXPORT_PATH.exists():
    EXPORT_PATH.mkdir()
if not IMAGE_PATH.exists():
    IMAGE_PATH.mkdir()
    print(f"Copy images in {IMAGE_PATH.resolve()}.")
if not LABEL_PATH.exists():
    LABEL_PATH.mkdir()

if not IMG_TRAINING_PATH.exists():
    IMG_TRAINING_PATH.mkdir()

if not IMG_VALIDATION_PATH.exists():
    IMG_VALIDATION_PATH.mkdir()

if not TXT_TRAINING_PATH.exists():
    TXT_TRAINING_PATH.mkdir()

if not TXT_VALIDATION_PATH.exists():
    TXT_VALIDATION_PATH.mkdir()

# Loading JSON
with open("labels.json") as f:
    labels_dict = json.loads(f.read())

file_names = []

# Extracting relevant values from the JSON and saving them in the YOLO format
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

        file_names.append(file_name)

# Generating the classes file for the YOLO format
with open(EXPORT_PATH.joinpath("classes.txt"), "w") as f:
    f.write("vocal_cords")

# Splitting the images into training and validation datasets
train_val_split = 0.8
total_count = len(file_names)
random_file_names = file_names.copy()
random.shuffle(random_file_names)

for i, name in enumerate(random_file_names):
    # Images
    old_path = Path(IMAGE_PATH, f"{name}.jpg")
    if not old_path.exists():
        print("Image in the images folder not found, terminating the code.")
        break
    if i <= int(train_val_split * total_count):
        new_path = IMG_TRAINING_PATH.joinpath(f"{name}.jpg")
    else:
        new_path = IMG_VALIDATION_PATH.joinpath(f"{name}.jpg")

    old_path.rename(new_path)

    # Labels
    old_path = Path(LABEL_PATH, f"{name}.txt")
    if not old_path.exists():
        print("Label in the label folder not found, terminating the code.")
        break
    if i <= int(train_val_split * total_count):
        new_path = TXT_TRAINING_PATH.joinpath(f"{name}.txt")
    else:
        new_path = TXT_VALIDATION_PATH.joinpath(f"{name}.txt")

    old_path.rename(new_path)

# Generating YAML file for the YOLO format
datasets_path = ".."+"".join([f"/{x}" for x in list(EXPORT_PATH.parts)])
train_path = "/".join([x for x in list(IMG_TRAINING_PATH.parts[1:])])
validation_path = "/".join([x for x in list(IMG_VALIDATION_PATH.parts[1:])])

with open(Path("yolov5", "data", "config.yaml"), "w") as f:
    for type, path in zip(["path", "train", "val"], [datasets_path, train_path, validation_path]):
        f.write(f"{type}: {path}\n")
    f.write("names:\n  0: vocal cords")
