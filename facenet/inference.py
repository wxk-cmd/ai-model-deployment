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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
#
def cosine_similarity_custom(vector1, vector2):
        vector1 = vector1[0]
        vector2 = vector2[0]
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        similarity = dot_product / (norm_vector1 * norm_vector2)
        return similarity
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
def get_face_descriptor(img_path):
      aligned = []
      x  = cv2.imread(img_path)
      startTime = time.time()
      x_aligned, _ = mtcnn(x, return_prob=True)
      endTime  = time.time()
      if x_aligned is not None:
            # print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
            # data = torch.transpose(x_aligned,0,1)
            # data = torch.transpose(data,1,2) 
            # img = np.array(data *  255)
            print(aligned)
      print("MTCNN cost time: ", (endTime - startTime) * 1000, "ms")
      aligned = torch.stack(aligned).to(device)
      embeddings = resnet(aligned).detach().cpu()
      print("resnet cost time: ", (time.time() - endTime) * 1000, "ms")
      return embeddings

def main():
   img_path1 = "/home/cpz/facenet/facenet-pytorch-master/data/data_faces_from_camera/person_1_zpc/img_face_1.jpg"
   img_path2 = "/home/cpz/facenet/facenet-pytorch-master/data/data_faces_from_camera/person_1_zpc/img_face_4.jpg"
   score1 = get_face_descriptor(img_path1)
   score2 = get_face_descriptor(img_path2)
   print(score1)
   print(cosine_similarity_custom(np.array(score1), np.array(score2)))


if __name__ == "__main__":
      main()