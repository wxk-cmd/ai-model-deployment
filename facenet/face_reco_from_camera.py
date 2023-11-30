
# 摄像头实时人脸识别 / Real-time face detection and recognition


import numpy as np
import cv2
import pandas as pd
import os
import time
import logging
from PIL import Image, ImageDraw, ImageFont
from models.inception_resnet_v1 import  InceptionResnetV1
from models.mtcnn import  MTCNN
import torch
from torchvision import transforms
import  csv
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
# detector = mtcnn.detect()
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
transform = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
])

def get_face(image):
      # 进行人脸检测
      boxes, _ = mtcnn.detect(image)
      # 在原始图像上绘制检测到的人脸框
      if boxes is not None:
          return  boxes

def get_face_descriptor(img, boxes):
      cv2.imwrite("test.jpg", img)
      # print(boxes)
      x_min, y_min, x_max,y_max = int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3])
      # print(img.shape)
      face = img[y_min : y_max+1, x_min : x_max+1, :]
      # print(face.shape)
      # cv2.imwrite("crop.jpg", face)
      face = Image.fromarray(face)
      image_tensor = transform(face).unsqueeze(0).to(device)
      embeddings = resnet(image_tensor).detach().cpu()

      return embeddings

class Face_Recognizer:
    def __init__(self):
        self.face_feature_known_list = []                # 用来存放所有录入人脸特征的数组 / Save the features of faces in database
        self.face_name_known_list = []                   # 存储录入人脸名字 / Save the name of faces in database

        self.current_frame_face_cnt = 0                     # 存储当前摄像头中捕获到的人脸数 / Counter for faces in current frame
        self.current_frame_face_feature_list = []           # 存储当前摄像头中捕获到的人脸特征 / Features of faces in current frame
        self.current_frame_face_name_list = []              # 存储当前摄像头中捕获到的所有人脸的名字 / Names of faces in current frame
        self.current_frame_face_name_position_list = []     # 存储当前摄像头中捕获到的所有人脸的名字坐标 / Positions of faces in current frame

        # Update FPS
        self.fps = 0                    # FPS of current frame
        self.fps_show = 0               # FPS per second
        self.frame_start_time = 0
        self.frame_cnt = 0
        self.start_time = time.time()

        self.font = cv2.FONT_ITALIC
        self.font_chinese = ImageFont.truetype("simsun.ttc", 30)

    # 从 "features_all.csv" 读取录入人脸特征 / Read known faces from "features_all.csv"
    # def get_face_database(self):
    #     if os.path.exists("data/features_all.csv"):
    #         path_features_known_csv = "data/features_all.csv"
    #         csv_rd = pd.read_csv(path_features_known_csv, header=None)
    #         for i in range(csv_rd.shape[0]):
    #             features_someone_arr = []
    #             self.face_name_known_list.append(csv_rd.iloc[i][0])
    #             for j in range(1, 129):
    #                 if csv_rd.iloc[i][j] == '':
    #                     features_someone_arr.append('0')
    #                 else:
    #                     features_someone_arr.append(csv_rd.iloc[i][j])
    #             self.face_feature_known_list.append(features_someone_arr)
    #         logging.info("Faces in Database: %d", len(self.face_feature_known_list))
    #         return 1
    #     else:
    #         logging.warning("'features_all.csv' not found!")
    #         logging.warning("Please run 'get_faces_from_camera.py' "
    #                         "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
    #         return 0

    def get_face_database(self):
        #  {
        #     'name' : name,
        #     'featureVector: list
        #  }
        # 返回一个数据库中的人的特征字典
            with open('features.csv', newline='') as csvfile:
                # 创建CSV文件的读取器对象
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    data  = row[1:]
                    for i in range(len(data)):
                        data[i] = float(data[i])
                    self.face_feature_known_list.append(dict(name=row[0], featureVector= data))
            csvfile.close()
            # print("数量",len(self.face_feature_known_list))
            return 1


    # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features
    @staticmethod
    def return_euclidean_distance(vector1, vector2):
            dot_product = np.dot(vector1, vector2)
            norm_vector1 = np.linalg.norm(vector1)
            norm_vector2 = np.linalg.norm(vector2)
            similarity = dot_product / (norm_vector1 * norm_vector2)
            return similarity

    # 更新 FPS / Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        # 每秒刷新 fps / Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    # 生成的 cv2 window 上面添加说明文字 / PutText on cv2 window
    def draw_note(self, img_rd):
        cv2.putText(img_rd, "Face Recognizer", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps_show.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_name(self, img_rd):
        # 在人脸框下面写人脸名字 / Write names under rectangle
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i in range(self.current_frame_face_cnt):
            # print("draw")

            # cv2.putText(img_rd, self.current_frame_face_name_list[i], self.current_frame_face_name_position_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            draw.text(xy=self.current_frame_face_name_position_list[i], text=self.current_frame_face_name_list[i], font=self.font_chinese,
                  fill=(255, 255, 0))
            img_rd = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_rd

    # 修改显示人名 / Show names in chinese
    def show_chinese_name(self):
        # Default known name: person_1, person_2, person_3
        if self.current_frame_face_cnt >= 1:
            # 修改录入的人脸姓名 / Modify names in face_name_known_list to chinese name
            self.face_name_known_list[0] = '张三'.encode('utf-8').decode()
            # self.face_name_known_list[1] = '张四'.encode('utf-8').decode()

    # 处理获取的视频流，进行人脸识别 / Face detection and recognition from input video stream
    def process(self, stream):
        # 1. 读取存放所有人脸特征的 csv / Read known faces from "features.all.csv"
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame %d starts", self.frame_cnt)
                flag, img_rd = stream.read()
                faces, _  = mtcnn.detect(img_rd)
                # print(faces)
                kk = cv2.waitKey(1)
                # 按下 q 键退出 / Press 'q' to quit
                if kk == ord('q'):
                    break
                else:
                    self.draw_note(img_rd)
                    self.current_frame_face_feature_list = []
                    self.current_frame_face_cnt = 0
                    self.current_frame_face_name_position_list = []
                    self.current_frame_face_name_list = []

                    # 2. 检测到人脸 / Face detected in current frame
                    if faces is not  None:
                        # 3. 获取当前捕获到的图像的所有人脸的特征 / Compute the face descriptors for faces in current frame
                        for i in range(len(faces)):
                            # shape = predictor(img_rd, faces[i])
                            # print("test",i)
                            self.current_frame_face_feature_list.append(get_face_descriptor(img_rd, faces[i]))
                        # 4. 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                        for k in range(len(faces)):
                            logging.debug("For face %d in camera:", k+1)
                            # 先默认所有人不认识，是 unknown / Set the default names of faces with "unknown"
                            self.current_frame_face_name_list.append("unknown")

                            # 每个捕获人脸的名字坐标 / Positions of faces captured
                            # self.current_frame_face_name_position_list.append(tuple(
                            #     [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_name_position_list.append(tuple(
                                [faces[k][0], int(faces[k][3] + (faces[k][3] - faces[k][1]) / 12 )]))

                            # 5. 对于某张人脸，遍历所有存储的人脸特征
                            # For every faces detected, compare the faces in the database
                            current_frame_e_distance_list = []
                            for i in range(len(self.face_feature_known_list)):
                                # print(self.face_feature_known_list[i])
                                if len(self.face_feature_known_list[i]['featureVector']) :
                                    e_distance_tmp = self.return_euclidean_distance(self.current_frame_face_feature_list[k],
                                                                                    self.face_feature_known_list[i]['featureVector'])
                                    logging.debug("  With person %s, the e-distance is %f", str(i + 1), e_distance_tmp)
                                    current_frame_e_distance_list.append(e_distance_tmp)
                                else:
                                    # 空数据 person_X
                                    current_frame_e_distance_list.append(999999999)
                            # 6. 寻找出最小的欧式距离匹配 / Find the one with minimum e-distance
                            similar_person_num = current_frame_e_distance_list.index(max(current_frame_e_distance_list))
                            logging.debug("Minimum e-distance with %s: %f", self.face_feature_known_list[similar_person_num]['name'], max(current_frame_e_distance_list))
                            if max(current_frame_e_distance_list) > 0.75:
                                print(e_distance_tmp)

                                self.current_frame_face_name_list[k] = self.face_feature_known_list[similar_person_num]['name']
                                logging.debug("Face recognition result: %s", self.face_feature_known_list[similar_person_num]['name'])
                            else:
                                logging.debug("Face recognition result: Unknown person")
                            logging.debug("\n")

                            # 矩形框 / Draw rectangle
                            for kk, d in enumerate(faces):
                                # 绘制矩形框
                                cv2.rectangle(img_rd, tuple([int(d[0]), int(d[1])]), tuple([int(d[2]), int(d[3])]),(255, 255, 255), 2)


                        self.current_frame_face_cnt = len(faces)

                        # 7. 在这里更改显示的人名 / Modify name if needed
                        # self.show_chinese_name()

                        # 8. 写名字 / Draw name
                        img_with_name = self.draw_name(img_rd)
                        print(self.current_frame_face_name_list)

                    else:
                        img_with_name = img_rd

                logging.debug("Faces in camera now: %s", self.current_frame_face_name_list)

                cv2.imshow("camera", img_with_name)

                # 9. 更新 FPS / Update stream FPS
                self.update_fps()
                logging.debug("Frame ends\n\n")

    # OpenCV 调用摄像头并进行 process
    def run(self):
        # cap = cv2.VideoCapture("video.mp4")  # Get video stream from video file
        cap = cv2.VideoCapture(0)              # Get video stream from camera
        cap.set(3, 1000)                        # 640x680
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    # logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()
