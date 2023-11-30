import argparse
import time
from pathlib import Path
from facenet.models.inception_resnet_v1 import InceptionResnetV1
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from numpy import random
import csv
import numpy
from config import camera_configs
from PIL import Image
from torchvision import transforms
from models.experimental import attempt_load
from models.common import Conv, DWConv
from utils.datasets import LoadStreams, MyLoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

threshold = 0.65
transform = transforms.Compose([
    transforms.Resize(160),
    transforms.ToTensor(),
    transforms.Normalize([0.656, 0.487, 0.411], [1., 1., 1.])
])

judge_result = None


def judges(model_name, det, names):
    for *xyxy, conf, cls in reversed(det):
        if names[int(cls)] in config.yolo_model_configs[model_name]["event"]:
            return names[int(cls)]

class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None

def load_model(opt):
    device = select_device(opt.device)
    model = Ensemble()
    ckpt = torch.load(opt.weights, map_location=device)  # load
    model.append(ckpt['ema' if ckpt.get('ema')
                 else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % opt.weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))

    if device.type != 'cpu':
        model.half()  # to FP16
    return model


def return_euclidean_distance(vector1, vector2):
    dot_product = numpy.dot(vector1, vector2)
    norm_vector1 = numpy.linalg.norm(vector1)
    norm_vector2 = numpy.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

# 根据传入的特征向量，与预先存储的人脸特征列表中的特征向量进行比较，并返回与之最相似的人脸对应的名称
def return_name(vector,face_feature_known_list):
    index = 0
    similar = 0
    for i in range(len(face_feature_known_list)):
        e_distance_tmp = return_euclidean_distance(
            vector, face_feature_known_list[i]['featureVector'])
        if similar < e_distance_tmp:
            index = i
            similar = e_distance_tmp

    if similar > threshold:
        return face_feature_known_list[index]['name']
    return "face"

# 从图像中提取人脸描述符的函数
def get_face_descriptor(img, boxes):
    x_min, y_min, x_max, y_max = int(boxes[0]), int(
        boxes[1]), int(boxes[2]), int(boxes[3])
    face = img[y_min: y_max + 1, x_min: x_max + 1, :]
    face = Image.fromarray(face)
    image_tensor = transform(face).unsqueeze(0).to(device)
    embeddings = resnet(image_tensor).detach().cpu()[0]
    return embeddings

def get_face_database(list_path):
    face_feature_known_list = []
    with open(list_path, "r", encoding="utf-8", newline='') as csvfile:
        # 创建CSV文件的读取器对象
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data = row[1:]
            for i in range(len(data)):
                data[i] = float(data[i])
            face_feature_known_list.append(
                dict(name=row[0], featureVector=data))
    csvfile.close()
    return face_feature_known_list


class Option:
    def __init__(self, weights,
                 img_size, conf_thres,
                 iou_thres=0.45, device='', view_img=False,
                 classes=None, agnostic_nms=False,
                 augment=False, update=False, exist_ok=False, **kwargs):
        self.weights = weights
        self.source = None
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.update = update
        self.exist_ok = exist_ok


class DetectAPI:
    def __init__(self, opt, model, model_name):
        self.opt = opt
        self.model_name = model_name

        # exit()
        # source, weights, view_img, save_txt, imgsz, trace = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size, not self.opt.no_trace
        # save_img = not self.opt.nosave and not self.source.endswith('.txt')  # save inference images
        # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        #     ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        # self.half=True
        # model
        self.model = model
        self.stride = 32  # int(self.model.stride.max())  # model stride

        imgsz = self.opt.img_size
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load(
                'weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # read names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]

    def detect(self, source):  # 使用时，调用这个函数
        # Set Dataloader
        vid_path, vid_writer = None, None
        if type(source) != list:
            raise TypeError(
                'source must be a list which contain  pictures read by cv2')

        dataset = MyLoadImages(source, img_size=self.imgsz,
                               stride=self.stride)  # imgsz
        # 原来是通过路径加载数据集的，现在source里面就是加载好的图片，所以数据集对象的实现要
        # 重写。修改代码后附。在utils.dataset.py上修改。

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        # t0 = time.time()
        result = []
        '''
        for path, img, im0s, vid_cap in dataset:'''
        for img, im0s in dataset:
            img = torch.from_numpy(img).to(self.device)
            # img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img = img.float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            # t1 = time_synchronized()
            pred = self.model(img, augment=self.opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            # t2 = time_synchronized()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)
                # Print time (inference + NMS)
                # print(f'{s}Done. ({t2 - t1:.3f}s)')
                # Process detections
            det = pred[0]  # 原来的情况是要保持图片，因此多了很多关于保持路径上的处理。另外，pred
            # 其实是个列表。元素个数为batch_size。由于对于我这个api，每次只处理一个图片，
            # 所以pred中只有一个元素，直接取出来就行，不用for循环。
            im0 = im0s.copy()  # 这是原图片，与被传进来的图片是同地址的，需要copy一个副本，否则，原来的图片会受到影响
            # s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            result_txt = []
            # 对于一张图片，可能有多个可被检测的目标。所以结果标签也可能有多个。
            # 每被检测出一个物体，result_txt的长度就加一。result_txt中的每个元素是个列表，记录着
            # 被检测物的类别引索，在图片上的位置，以及置信度
            global judge_result
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                
                for *xyxy, conf, cls in reversed(det):
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # if model_type == "face":
                    #     face_feature = get_face_descriptor(im0, [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                    #     name = retrun_name(face_feature)
                    line = (int(cls.item()), [int(_.item())
                            for _ in xyxy], conf.item())  # label format
                    result_txt.append(line)
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    # plot_one_box(xyxy, im0, label=label,
                    #              color=self.colors[int(cls)], line_thickness=3)

            
                judge_result = judges(self.model_name, det, self.names)

            result.append((im0, result_txt))  # 对于每张图片，返回画完框的图片，以及该图片的标签列表。

        return result, self.names, judge_result
