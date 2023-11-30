import cv2
from models.inception_resnet_v1 import  InceptionResnetV1
from models.mtcnn import  MTCNN
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
#
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
def cosine_similarity_custom(vector1, vector2):
        vector1 = vector1[0]
        vector2 = vector2[0]
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        similarity = dot_product / (norm_vector1 * norm_vector2)
        return similarity
# def collate_fn(x):
#     return x[0]

# dataset = datasets.ImageFolder("/home/cpz/facenet/facenet-pytorch-master/data/data_faces_from_camera")
# dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
# loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

# aligned = []
# names = []
# for x, y in loader:
# #     print(x,x.size(), type(x))
#     print(x)
#     x_aligned, prob = mtcnn(x, return_prob=True)
#     if x_aligned is not None:
#         print('Face detected with probability: {:8f}'.format(prob))
#         aligned.append(x_aligned)
#         data = torch.transpose(x_aligned,0,1)
#         data = torch.transpose(data,1,2)
#         img = np.array(data *  255)
#         print(img.shape)

#         cv2.imwrite(f"{prob}.jpg", img)
#         names.append(dataset.idx_to_class[y])

# aligned = torch.stack(aligned).to(device)
# data = torch.rand(1,3,180,160).cuda()
# embeddings = resnet(data).detach().cpu()


# print(embeddings.shape)

image1 =  Image.open("/home/cpz/facenet/facenet-pytorch-master/testt.jpg")
image2 =  Image.open("/home/cpz/facenet/facenet-pytorch-master/testtt.jpg")
image3 =  Image.open("/home/cpz/facenet/facenet-pytorch-master/test0.jpg")

transform = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
])

# 3. 应用转换操作
image_tensor1 = transform(image1).unsqueeze(0).cuda()
image_tensor2 = transform(image2).unsqueeze(0).cuda()
image_tensor3 = transform(image3).unsqueeze(0).cuda()
embeddings1 = resnet(image_tensor1).detach().cpu()
embeddings2 = resnet(image_tensor2).detach().cpu()
embeddings3 = resnet(image_tensor3).detach().cpu()
print(cosine_similarity_custom(embeddings1, embeddings2))
print(cosine_similarity_custom(embeddings1, embeddings3))
print(cosine_similarity_custom(embeddings2, embeddings3))

