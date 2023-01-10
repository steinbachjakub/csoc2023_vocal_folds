from pathlib import Path
import shutil
import cv2

# Set Path variables for videos and frames
PATH_VIDEO = Path(".", "videos")
PATH_FRAMES = Path(".", "frames")
INITIAL_FRAMES = [260, 173, 165]
FINAL_FRAMES = [1064, 839, 425]
# Check if there are frames or not and clear the directory
if PATH_FRAMES.is_dir():
    shutil.rmtree(PATH_FRAMES)
# Create the directory for all files
PATH_FRAMES.mkdir()
for file in PATH_VIDEO.iterdir():
    PATH_FRAMES.joinpath(file.name.split(".")[0]).mkdir()

for file, initial_frame, final_frame in zip(PATH_VIDEO.iterdir(), INITIAL_FRAMES, FINAL_FRAMES):
    print(f"Working on {file.name}...")

    # Read the video from specified path
    cam = cv2.VideoCapture(str(file))

    for current_frame in range(initial_frame, final_frame + 1):
        # reading from frame
        cam.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
        _, frame = cam.read()
        # if video is still left continue creating images
        frame_name = PATH_FRAMES.joinpath(file.name.split(".")[0], f"{file.name.split('.')[0]}_frame_{current_frame}.jpg")
        print(f'Creating... {frame_name.name}')
        # writing the extracted images
        cv2.imwrite(str(frame_name), frame)
        # increasing counter so that it will
        # show how many frames are created
        current_frame += 1

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
