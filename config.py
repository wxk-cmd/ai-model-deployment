# 通过 django 后端接口返回来的示例
# 模型配置
yolo_model_configs = {
    "phone": {  # 模型名称
        "weights": "/media/vision/D/ai-model-deployment/Model/phone_best.pt",
        "img_size": 416,
        "conf_thres": 0.5,
        "event": "phone"  #
    },
    "safetyhat": {
        "weights": "/media/vision/D/ai-model-deployment/Model/safety_hemlt.pt",
        "img_size": 640,
        "conf_thres": 0.5,
        "event": "head"  #
    },
    "smoking": {
        "weights": "/media/vision/D/ai-model-deployment/Model/smoking.pt",
        "img_size": 416,
        "conf_thres": 0.2,
        "event": "smoking"  #
    },
    "face": {
        "weights": "/media/vision/D/ai-model-deployment/Model/face_best.pt",
        "img_size": 640,
        "conf_thres": 0.15,
        "event": "face"
    }
}
resnet_model_configs = {
    # "fire": {
    #     "weights": "/home/vision/Desktop/ai-model-deployment/fire_smoking_model/ResNet34_fire_dataset.pth",
    #     "json": "/home/vision/Desktop/ai-model-deployment/fire_smoking_model/fire_dataset_class_indices.json",
    #     "img_size": 256,
    # },
    # "somke": {
    #     "weights": "/home/vision/Desktop/ai-model-deployment/fire_smoking_model/ResNet34_smoke_dataset.pth",
    #     "json": "/home/vision/Desktop/ai-model-deployment/fire_smoking_model/smoke_dataset_class_indices.json",
    #     "img_size": 256,
    # },
}

# 摄像头配置
camera_configs = {
    "1": {  # 摄像头名称
        "camera_name": "1",
        "url": "rtsp://admin:wspw123456@192.168.0.10/Streaming/Channels/1",
        "models": {
            "safetyhat": {  # "safetyhat" => 模型名称
                "detect_interval": 1,  # 单位：秒
            },
            "phone": {
                "detect_interval": 1,  # 单位：秒
            },
            "fire": {
                "detect_interval": 1,  # 单位：秒
            },
            "smoking": {
                "detect_interval": 1,  # 单位：秒
            },
            "face": {
                "detect_interval": 1,  # 单位：秒
                "face_feature_config": {
                    "face_path": '/home/vision/Desktop/ai-model-deployment/facenet/features.csv',  # 人脸的特征文件存储路径
                    # "type": "white",         # type是white 代表的是白名单
                    "type": "black"
                }
            }
        }
    },
    "2": {  # 摄像头名称
        "camera_name": "camera2",
        "url": "rtsp://admin:wspw123456@192.168.0.11/Streaming/Channels/1",
        # "url": 0,
        "models": {
            "safetyhat": {  # "safetyhat" => 模型名称
                "detect_interval": 1,  # 单位：秒
            },
            "phone": {
                "detect_interval": 1,  # 单位：秒
            },
            "fire": {
                "detect_interval": 1,  # 单位：秒
            },
            "smoking": {
                "detect_interval": 1,  # 单位：秒
            },
            # "face": {
            #     "detect_interval": 1,  # 单位：秒
            #     "face_feature_config": {
            #         "face_path": '',
            #         # "type": "white",
            #         "type": "black"
            #     }
            # }
        }
    },
    # "2": {
    #     "camera_name": "2",
    #     "url": "0",
    #     "models": {
    #         "phone": {
    #             "detect_interval": 1,  # 单位：秒
    #         }
    #     }
    # }
}
