from pathlib import Path
import glob
import numpy as np
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# PROJECT AND PARENT FOLDERS
PROJECT_FOLDER = Path().absolute()
PARENT_FOLDER = PROJECT_FOLDER.parent

# LABELED IMAGE FOLDERS
IMPORT_PATH = Path("labeled_images")
IMAGES_IMPORT = IMPORT_PATH.joinpath("images")
LABELS_IMPORT = IMPORT_PATH.joinpath("labels.json")

flag = True
def structure_checker():
    # YOLO SUBDIRECTORY
    yolo_search = glob.glob("**/*yolo*/train.py", root_dir=PARENT_FOLDER)
    if len(yolo_search) == 0:
        print("YOLOv5 not found in the project or parent directories. \n"
              "Please, follow https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data to download YOLOv5.")
        flag = False
    elif len(yolo_search) == 1:
        YOLO_TRAIN = Path(yolo_search[0]).absolute()
        print(f"YOLOv5 found at {YOLO_TRAIN.relative_to(PARENT_FOLDER).parent}.")
    else:
        print("Issue with multiple YOLOv5 folders in the project or parent directories.")
        flag = False

    # RAW DATA SUBDIRECTORY
    if IMAGES_IMPORT.exists():
        print(f"Found raw data at {IMAGES_IMPORT}.")
    else:
        IMAGES_IMPORT.mkdir(parents=True)
        print(f"Raw data not found. Please, copy your images to {IMAGES_IMPORT} "
              f"and your labels to {LABELS_IMPORT}.")
        flag = False

    # DATASETS SUBDIRECTORY


if __name__ == "__main__":
    structure_checker()
    # name = "89c35c7c-2_pareza_frame_265"
    # img = mpimg.imread("0c0ac43e-fyziologicky_nalez_frame_312.jpg")
    # with open("0c0ac43e-fyziologicky_nalez_frame_312.txt", "r") as f:
    #     _, x_cr, y_cr, w_r, h_r = f.read().split(" ")
    #     print(f.read())
    # x_cr = float(x_cr)
    # y_cr = float(y_cr)
    # w_r = float(w_r)
    # h_r = float(h_r)
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(img)
    # rect = Rectangle(((x_cr - w_r/2)*960, (y_cr - h_r/2)*720), w_r*960, h_r*720, fill=None)
    # ax.add_patch(rect)
    # plt.show()

    # datasets = ["first", "kfolded"][np.random.randint(0, 2)]
    datasets = "kfolded"
    # set_type = ["training", "validation"][np.random.randint(0, 2)]
    set_type = "validation"
    print(datasets)
    if datasets == "kfolded":
        random_fold = np.random.randint(0, 5)
        img_path = Path("datasets", "images", f"fold{random_fold}", set_type)
        label_path = Path("datasets", "labels", f"fold{random_fold}", set_type)
    else:
        img_path = Path(f"C:/Users/steinbaj/Desktop/datasets/images/{set_type}")
        label_path = Path(f"C:/Users/steinbaj/Desktop/datasets/labels/{set_type}")

    print(img_path)
    print(label_path)
    img_list = list(img_path.iterdir())
    img_count = len(img_list)
    rand_indices = np.random.randint(0, img_count - 1, 36)
    figsize = np.array((9.6, 7.2 + 1)) * 1.5

    fig, axs = plt.subplots(6, 6, figsize=figsize)
    fig.suptitle(f"Using images from {img_path}")
    axs = np.ravel(axs)

    for i, index in enumerate(rand_indices):
        name = label_path.joinpath(img_list[index].name.split(".")[0] + ".txt")
        img = mpimg.imread(img_list[index])

        _, xcr, ycr, wr, hr = np.loadtxt(name, delimiter=" ")

        x = (xcr - wr / 2) * 960
        y = (ycr - hr / 2) * 720
        w = wr * 960
        h = hr * 720

        rect = Rectangle((x, y), w, h, fill=None, edgecolor="yellow")
        axs[i].imshow(img)
        axs[i].add_patch(rect)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.subplots_adjust(0,0,1,0.95,0,0)
    plt.show()