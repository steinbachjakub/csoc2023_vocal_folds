from pathlib import Path
import shutil
import cv2

# Set Path variables for videos and frames
PATH_VIDEO = Path(".", "videos")
PATH_FRAMES = Path(".", "frames")
PATH_FRAMES_GOOD = Path(".", "frames", "good")
PATH_FRAMES_BAD = Path(".", "frames", "bad")
INITIAL_FRAMES = [260, 173, 165]
FINAL_FRAMES = [1064, 839, 425]
# Check if there are frames or not and clear the directory
if PATH_FRAMES.is_dir():
    shutil.rmtree(PATH_FRAMES)
# Create the directory for all files
PATH_FRAMES.mkdir()
PATH_FRAMES_GOOD.mkdir()
PATH_FRAMES_BAD.mkdir()
for file in PATH_VIDEO.iterdir():
    for path in [PATH_FRAMES_GOOD, PATH_FRAMES_BAD]:
        path.joinpath(file.name.split(".")[0]).mkdir()

for file, initial_frame, final_frame in zip(PATH_VIDEO.iterdir(), INITIAL_FRAMES, FINAL_FRAMES):
    print(f"Working on {file.name}...")

    # Read the video from specified path
    cam = cv2.VideoCapture(str(file))

    current_frame = 1
    while True:
        # reading from frame
        cam.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
        ret, frame = cam.read()
        # Stopping the loop in case the end of the video is reached
        if not ret:
            break
        if (current_frame >= initial_frame) and (current_frame <= final_frame):
            # place it in the folder with good images
            frame_name = PATH_FRAMES_GOOD.joinpath(file.name.split(".")[0],
                                              f"{file.name.split('.')[0]}_frame_{current_frame}.jpg")
        else:
            # place it in the folder with bad images
            frame_name = PATH_FRAMES_BAD.joinpath(file.name.split(".")[0],
                                              f"{file.name.split('.')[0]}_frame_{current_frame}.jpg")
        # print(f'Creating... {frame_name.name}')
        # writing the extracted images
        cv2.imwrite(str(frame_name), frame)
        # increasing counter so that it will
        # show how many frames are created
        current_frame += 1
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
