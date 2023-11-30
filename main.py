import cv2
import torch
import threading
import time
import os
import requests
from utils.plots import plot_one_box
from detect_api import DetectAPI, Option, load_model, get_face_descriptor, return_name, get_face_database
import config
from resnet_predict_api import load_models, ResnetDetectAPI

import requests
import get_config
from db import MySQLInserter
import get_config
import model_config
import camera_config

from logger import logger


# 在全局范围内定义一个变量来保存图像
global_image_to_show = None
global_image_lock = threading.Lock()
out_ip = 'http://192.168.0.203:8000/static/images/'
url = 'http://127.0.0.1:8000/api/insertAlarm'



# resnet
def resnet_handle_detections(frame, result, output_folder, camera_name, model_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    detection_info = {}
    now_time = time.strftime('%Y%m%d_%H%M%S')
    if 'no' not in result:
        detection_info = {
            'camera_name': camera_name,
            'model_name': model_name,
            'detection_time': now_time,
            'result': result,
            'image_save_path': '',

        }
    else:
        return
    output_path = os.path.join(
        output_folder, f"{camera_name}_{model_name}_{now_time}.jpg")
    if detection_info:
        cv2.imwrite(output_path, frame)
        out_ips = out_ip + \
            f"{camera_name}_{model_name}_{now_time}.jpg"
        detection_info['image_save_path'] = out_ips

        # inserter.insert_data(TABLE_NAME, detection_info)
        # response = requests.post(url, json=detection_info)
        # if response.status_code == 200:
        #     print("success")
    if detection_info != {}:
        logger.info(model_name + "检测结果")
        logger.info(detection_info)
    print(model_name + "检测结果")
    print(detection_info)


def handle_detections(frame, detections, names, output_folder, camera_name, model_name, result):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 保存检测到的对象的信息
    detection_info = {}
    if model_name in camera_config.camera_config[camera_name]["models"] and result:
        now_time = time.strftime('%Y%m%d_%H%M%S')
        for _, (im, det) in enumerate(detections):
            if len(det):
                for cls, xyxy, conf in det:
                    if model_name == "face":
                        face_config = camera_config.camera_config[camera_name]["models"]["face"]
                        if "face_feature_config" in face_config:
                            face_feature_config = face_config["face_feature_config"]
                            face_path = face_feature_config.get("face_path")
                            face_type = face_feature_config.get("type")
                            face_feature_known_list = get_face_database(face_path)
                            face_feature = get_face_descriptor(im, [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                            name = return_name(face_feature,face_feature_known_list)
                            label = f'{names[int(cls)]} {conf:.2f}'
                            if name != config.yolo_model_configs[model_name]["event"]:
                                if face_type == "black":
                                    plot_one_box(xyxy, im, label=name, color=(
                                        0, 0, 255), line_thickness=2)
                                if face_type == "white":
                                    plot_one_box(xyxy, im, label=name, color=(
                                        255, 0, 0), line_thickness=2)
                                detection_info = {
                                    'camera_name': camera_name,
                                    'model_name': model_name,
                                    'detection_time': now_time,
                                    'result': name,
                                    'image_save_path': '',
                                }
                            else:
                                plot_one_box(xyxy, im, label=label, color=(
                                    0, 255, 0), line_thickness=2)

                    else:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        if label == config.yolo_model_configs[model_name]["event"]:
                            plot_one_box(xyxy, im, label=label, color=(
                                0, 0, 255), line_thickness=2)
                        else:
                            plot_one_box(xyxy, im, label=label, color=(
                                0, 255, 0), line_thickness=2)
                        detection_info = {
                            'camera_name': camera_name,
                            'model_name': model_name,
                            'detection_time': now_time,
                            'result': result,
                            'image_save_path': '',
                            # 'confidence': f'{conf:.2f}',
                            # 'bbox': [int(x) for x in xyxy]

                        }
        # 保存检测结果为图片,检测到了再保存
        output_path = os.path.join(
            output_folder, f"{camera_name}_{model_name}_{now_time}.jpg")
        if detection_info:
            cv2.imwrite(output_path, im)
            out_ips = out_ip + \
                f"{camera_name}_{model_name}_{now_time}.jpg"
            detection_info['image_save_path'] = out_ips

            # inserter.insert_data(TABLE_NAME, detection_info)
            response = requests.post(url, json=detection_info)
            if response.status_code == 200:
                print("success")

    
    if detection_info != {}:
        logger.info(model_name + "检测结果")
        logger.info(detection_info)
    print(model_name + "检测结果")
    print(detection_info)

    # 将检测到的对象的信息保存到文本文件
    # with open(output_path.replace('.jpg', '.txt'), 'w') as f:
    #     for info in detection_info:
    #         f.write(f"Class: {info['class']}, Confidence: {info['confidence']}, BBox: {info['bbox']}\n")


def rotate_frame(frame, angle: int):
    """旋转frame角度

    Args:
        frame (_type_): 摄像头帧
        angle (int): 旋转角度，正数为顺时针旋转，负数为逆时针旋转

    Returns:
        _type_: 旋转后的frame
    """
    # 获取图像尺寸
    height, width = frame.shape[:2]

    # 计算旋转中心
    center = (width / 2, height / 2)

    # 使用cv2.getRotationMatrix2D获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # 使用cv2.warpAffine应用旋转
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))

    return rotated_frame


def process_camera(camera_config, RESNET_API_SELECT, YOLO_API_SELECT):
    global global_image_to_show

    cap = cv2.VideoCapture(camera_config["url"])
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(fps)

    reset_interval_frames = 10 * 60 * fps  # 每10分钟清零
    frame_count = 0

    last_frame_count = {}
    output_folder = "/media/vision/D/box-python/backend/static/images"  # 输出文件夹
    # output_folder = "results"

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 旋转角度
        frame = rotate_frame(frame, camera_config["angle"])

        with global_image_lock:
            global_image_to_show = frame

        frame_count += 1
        if frame_count >= reset_interval_frames:
            frame_count = 0
        for model_name in camera_config["models"]:
            last_frame_count[model_name] = 0
        for model_name, model_conf in camera_config["models"].items():
            detect_interval = model_conf["detect_interval"]
            frames_interval = int(detect_interval * fps)  # 根据摄像头fps换算抽帧间隔

            if model_name in RESNET_API_SELECT and frame_count % frames_interval == 0:
                # (model_name not in last_frame_count or (frame_count - last_frame_count[model_name] >= frames_interval)):
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = RESNET_API_SELECT[model_name].resnet_dectect(frame)
                resnet_handle_detections(
                    frame, result, output_folder, camera_config['camera_name'], model_name)

            if model_name in YOLO_API_SELECT and frame_count % frames_interval == 0:
                # (model_name not in last_frame_count or (frame_count - last_frame_count[model_name] >= frames_interval)):
                detections, names, result = YOLO_API_SELECT[model_name].detect([
                                                                               frame])
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 处理检测结果
                handle_detections(frame, detections, names, output_folder, camera_config['camera_name'],
                                  model_name, result)

                # 显示结果
                last_frame_count[model_name] = frame_count

    cap.release()


if __name__ == '__main__':
    # yolo
    YOLO_API_SELECT = {}

    # yolo_model_configs = config.yolo_model_configs
    # model_url= "http://192.168.0.203:8000/api/algorithm_managementList"
    # get_config.get_model_config(model_url)
    yolo_model_configs = model_config.model_config
    for model_name, conf in yolo_model_configs.items():
        opt = Option(**conf)
        model = load_model(opt)
        logger.info(model_name + "模型加载成功...")
        print(model_name + "模型加载成功...")
        YOLO_API_SELECT[model_name] = DetectAPI(
            opt=opt, model=model, model_name=model_name)

    # resnet
    RESNET_API_SELECT = {}
    resnet_model_configs = config.resnet_model_configs
    for model_name, conf in resnet_model_configs.items():
        data_transform, class_indict, device, model = load_models(
            conf["json"], conf["weights"], conf["img_size"])
        logger.info(model_name + "模型加载成功")
        print(model_name + "模型加载成功")
        RESNET_API_SELECT[model_name] = ResnetDetectAPI(model=model, data_transform=data_transform,
                                                        class_indict=class_indict, device=device)

    # camera_configs = config.camera_configs
    # camera_url = "http://192.168.0.203:8000/api/allEquipments"
    # get_config.get_camera_config(camera_url)  #up
    print(camera_config.camera_config)
    camera_configs=camera_config.camera_config


    threads = []
    for cam_name, cam_conf in camera_configs.items():
        t = threading.Thread(target=process_camera, args=(
            cam_conf, RESNET_API_SELECT, YOLO_API_SELECT))
        threads.append(t)
        t.start()

    # 在主线程中显示图像
    # while True:
        # if global_image_to_show is not None:
        #     with global_image_lock:
        #         cv2.imshow('Camera', global_image_to_show)
        # if cv2.waitKey(1) == ord('q'):
        #     break
    cv2.destroyAllWindows()

    for t in threads:
        t.join()
