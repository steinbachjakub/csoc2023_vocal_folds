import os
os.system("python yolov5/train.py --img 640 --batch 16 --epochs 100 --data ./datasets/config/config_0.yaml --weights yolov5s.pt")