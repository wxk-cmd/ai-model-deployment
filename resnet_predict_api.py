import cv2
import torch
import torchvision.transforms as transforms
import json
from PIL import Image
import os
import time
from resnet_model import resnet34


def load_models(dataset_json_path, dataset_weights_path, img_size):
    # 加载模型和相关配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(dataset_json_path, 'r') as f:
        class_indict = json.load(f)

    assert os.path.exists(dataset_weights_path), "file:'{}' does not exist.".format(dataset_weights_path)

    model = resnet34(num_classes=2).to(device)
    #model.load_state_dict(torch.load(dataset_weights_path))
    model.eval()

    return data_transform, class_indict, device, model


def get_image(data_transform, class_indict, model, device, frame_interval,url):
    # 初始化

    cap = cv2.VideoCapture(url)
    frame_count = 0

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()

        if not ret:
            break

        # 增加帧计数
        frame_count += 1

        # 如果时间间隔大于等于1秒，执行检测
        if frame_count % frame_interval == 0:
            # 转换帧为PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fire_or_somking_dectect(data_transform, class_indict, model, device, frame_rgb)

        # 显示摄像头帧
        cv2.imshow('Video Frame', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头和关闭窗口
    cap.release()
    cv2.destroyAllWindows()


class ResnetDetectAPI:
    def __init__(self, model, data_transform, class_indict, device):
        self.model = model
        self.data_transform = data_transform
        self.class_indict = class_indict
        self.device = device

    def resnet_dectect(self, frame_rgb):
        # 图像预处理
        img_pil = Image.fromarray(frame_rgb)
        img = self.data_transform(img_pil)
        img = torch.unsqueeze(img, dim=0)

        # 预测
        with torch.no_grad():
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            result = self.class_indict['1'] if (predict_cla > 0) else self.class_indict['0']
            print_res = "class:{} prob: {:.3}".format(self.class_indict[str(predict_cla)], predict[predict_cla].numpy())
            print(result)
        return result

def fire_or_somking_dectect(data_transform, class_indict, model, device, frame_interval):
    cap = cv2.VideoCapture(0)
    # 初始化
    frame_count = 0

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()

        if not ret:
            break

        # 增加帧计数
        frame_count += 1

        # 如果时间间隔大于等于1秒，执行检测
        if frame_count % frame_interval == 0:
            # 转换帧为PIL Image，并进行预处理
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img = data_transform(img_pil)
            img = torch.unsqueeze(img, dim=0)

            # 使用模型进行检测
            with torch.no_grad():
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()
                result = class_indict['1'] if (predict_cla > 0) else class_indict['0']
                print("预测类别：", result)
                print()
                print_res = "class:{} prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())

        # 显示摄像头帧
        cv2.imshow('Video Frame', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头和关闭窗口
    cap.release()
    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     weights="/home/vision/Desktop/ai-model-deployment/fire_smoking_model/ResNet34_fire_dataset.pth"
#     dataset_json="/home/vision/Desktop/ai-model-deployment/fire_smoking_model/fire_dataset_class_indices.json"
#     img_size = 256
#     url = "rtsp://admin:wspw123456@192.168.0.10/Streaming/Channels/1",
#     ddata_transform, class_indict, device, model = load_models(weights,dataset_json,img_size)
#     get_image(data_transform, class_indict, device, model,device,30,url)
