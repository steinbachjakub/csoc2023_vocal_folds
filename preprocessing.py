from pathlib import Path
import json
from random import shuffle
import shutil
import numpy as np

# Number of folds
K = 5

# Paths
IMPORT_PATH = Path("labeled_images")
IMAGES_IMPORT = IMPORT_PATH.joinpath("images")
LABELS_IMPORT = IMPORT_PATH.joinpath("labels.json")

EXPORT_PATH = Path("datasets")
IMAGES_EXPORT = EXPORT_PATH.joinpath("images")
LABELS_EXPORT = EXPORT_PATH.joinpath("labels")
CONFIG_EXPORT = EXPORT_PATH.joinpath("config")
CLASSES_EXPORT = EXPORT_PATH.joinpath("classes.txt")


# Deleting existing dataset
if EXPORT_PATH.exists():
    shutil.rmtree(EXPORT_PATH)

# Creating the directory
EXPORT_PATH.mkdir()
LABELS_EXPORT.mkdir()
CONFIG_EXPORT.mkdir()
IMAGES_EXPORT.mkdir()

# Loading JSON
with open(LABELS_IMPORT) as f:
    labels_dict = json.loads(f.read())

# Extracting relevant values from the JSON and saving them as dictionary under the correct name
labels = {} # labels[name] = (class, x, y, w, h)
file_names = []
for item in labels_dict:
    if len(item["annotations"][0]["result"]) > 0:
        img_width = item["annotations"][0]["result"][0]["original_width"]
        img_height = item["annotations"][0]["result"][0]["original_height"]
        x_coord = item["annotations"][0]["result"][0]["value"]["x"]
        y_coord = item["annotations"][0]["result"][0]["value"]["y"]
        frame_width = item["annotations"][0]["result"][0]["value"]["width"]
        frame_height = item["annotations"][0]["result"][0]["value"]["height"]
        file_name_full = item["file_upload"]
        file_name = item["file_upload"].split(".")[0]
        center_x = 1/100 * (x_coord + frame_width / 2)
        center_y = 1/100 * (y_coord + frame_height / 2)
        width = frame_width / 100
        height = frame_height / 100

        labels[file_name_full] = (0, center_x, center_y, width, height)

        # with open(LABELS_EXPORT.joinpath(f"{file_name}.txt"), "w") as f:
        #     f.write(f"0 {center_x} {center_y} {width} {height}")

        file_names.append(file_name_full)
    else:
        print(item["file_upload"].split(".")[0])

# Generating the classes file for the YOLO format
with open(EXPORT_PATH.joinpath("classes.txt"), "w") as f:
    f.write("vocal_cords")

# Splitting the images into training and validation datasets
total_count = len(file_names)
# k_in_fold = [(total_count // K) + 1] * (total_count % K) + [total_count // K] * (K - (total_count % K))
fold_number = np.linspace(1, total_count, total_count, dtype=np.int16) % K
random_file_names = np.array(file_names)
shuffle(random_file_names)

for i in range(K):
    FOLD_TRAINING = Path(f"fold{i}", "training")
    FOLD_VALIDATION = Path(f"fold{i}", "validation")

    IMAGES_EXPORT.joinpath(FOLD_TRAINING).mkdir(parents=True)
    IMAGES_EXPORT.joinpath(FOLD_VALIDATION).mkdir(parents=True)
    LABELS_EXPORT.joinpath(FOLD_TRAINING).mkdir(parents=True)
    LABELS_EXPORT.joinpath(FOLD_VALIDATION).mkdir(parents=True)

    img_training_set = [IMAGES_IMPORT.joinpath(x) for x in random_file_names[fold_number != i]]
    img_validation_set = [IMAGES_IMPORT.joinpath(x) for x in random_file_names[fold_number == i]]

    for file in img_training_set:
        name = file.name
        shutil.copy(file, IMAGES_EXPORT.joinpath(FOLD_TRAINING).joinpath(file.name))
        with open(LABELS_EXPORT.joinpath(FOLD_TRAINING).joinpath(f"{file.name.split('.')[0]}.txt"), "w") as f:
            f.write("0\t{}\t{}\t{}\t{}".format(labels[name][0], labels[name][1], labels[name][2], labels[name][3]))
    for file in img_validation_set:
        shutil.copy(file, IMAGES_EXPORT.joinpath(FOLD_VALIDATION).joinpath(file.name))
        with open(LABELS_EXPORT.joinpath(FOLD_VALIDATION).joinpath(f"{file.name.split('.')[0]}.txt"), "w") as f:
            f.write("0\t{}\t{}\t{}\t{}".format(labels[name][0], labels[name][1], labels[name][2], labels[name][3]))

    with open(CONFIG_EXPORT.joinpath(f"config_{i}.yaml"), "w") as f:
        f.write(f"path: {EXPORT_PATH.absolute()}\n")
        f.write(f"train: {IMAGES_EXPORT.joinpath(FOLD_TRAINING).relative_to(EXPORT_PATH)}\n")
        f.write(f"val: {IMAGES_EXPORT.joinpath(FOLD_VALIDATION).relative_to(EXPORT_PATH)}\n")
        f.write(f"names:\n  0: vocal_cords")

print(img_validation_set)
# for i, name in enumerate(random_file_names):
#     # Images
#     old_path = Path(IMAGE_PATH, f"{name}.jpg")
#     if not old_path.exists():
#         print("Image in the images folder not found, terminating the code.")
#         break
#     if i <= int(train_val_split * total_count):
#         new_path = IMG_TRAINING_PATH.joinpath(f"{name}.jpg")
#     else:
#         new_path = IMG_VALIDATION_PATH.joinpath(f"{name}.jpg")
#
#     old_path.rename(new_path)
#
#     # Labels
#     old_path = Path(LABEL_PATH, f"{name}.txt")
#     if not old_path.exists():
#         print("Label in the label folder not found, terminating the code.")
#         break
#     if i <= int(train_val_split * total_count):
#         new_path = TXT_TRAINING_PATH.joinpath(f"{name}.txt")
#     else:
#         new_path = TXT_VALIDATION_PATH.joinpath(f"{name}.txt")
#
#     old_path.rename(new_path)
#
# # Generating YAML file for the YOLO format
# datasets_path = ".."+"".join([f"/{x}" for x in list(EXPORT_PATH.parts)])
# train_path = "/".join([x for x in list(IMG_TRAINING_PATH.parts[1:])])
# validation_path = "/".join([x for x in list(IMG_VALIDATION_PATH.parts[1:])])
#
# with open(Path("yolov5", "data", "config.yaml"), "w") as f:
#     for type, path in zip(["path", "train", "val"], [datasets_path, train_path, validation_path]):
#         f.write(f"{type}: {path}\n")
#     f.write("names:\n\t: vocal cords")
