import cv2
from models.inception_resnet_v1 import  InceptionResnetV1
from models.mtcnn import  MTCNN
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import time
from PIL import Image, ImageDraw

# 初始化MTCNN模型
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
)

# 加载图像
image_path = r"C:\Users\ZPC\Desktop\facenet\facenet-pytorch-master\data\data_faces_from_camera\zpc\img_face_1.jpg"  # 替换为要检测的图像文件路径
image = Image.open(image_path)

# 进行人脸检测
boxes, _ = mtcnn.detect(image)

# 在原始图像上绘制检测到的人脸框
if boxes is not None:
    draw = ImageDraw.Draw(image)
    for box in boxes:
        print(box)
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=3)
        face = image.crop(box)
        face.save("testtt.jpg")
    
    # 保存带有人脸框的图像
    image.save(r"C:\Users\ZPC\Desktop\facenet\facenet-pytorch-master\data\data_faces_from_camera\zpc\img_face_1_zpctt.jpg")

# 如果要提取人脸，可以使用MTCNN.extract方法




# # 初始化MTCNN模型
# mtcnn = MTCNN()

# # 加载图像
# image_path = "/home/cpz/facenet/facenet-pytorch-master/data/data_faces_from_camera/person_1_zpc/img_face_1.jpg"  # 替换为要检测的图像文件路径
# image = Image.open(image_path)
# boxes, _ = mtcnn.detect(image)
# # 使用MTCNN提取人脸
# faces = mtcnn.extract(image, boxes,save_path = f"test{0}.jpg")

# # faces是一个包含提取的人脸的Tensor
# # 如果没有检测到人脸，faces将为None
# if faces is not None:
#     for i, face in enumerate(faces):
#         # 保存提取的人脸
#         save_path = f"test{i}.jpg"  # 替换为保存路径和文件名
#         Image.fromarray(face.permute(1, 2, 0).int().numpy()).save(save_path)
