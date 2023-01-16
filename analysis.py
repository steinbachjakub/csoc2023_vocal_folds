import os
os.system("python yolov5/train.py --img 640 --batch 16 --epochs 100 --data config.yaml --weights yolov5s.pt")