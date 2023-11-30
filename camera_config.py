camera_config={
    "76": {
        "camera_name": "76",
        "angle": 0,
        "url": "rtsp://admin:wspw123456@192.168.0.11/Streaming/Channels/1",
        "models": {
            "smoking": {
                "detect_interval": 4
            },
            "phone": {
                "detect_interval": 4
            },
            "safetyhat": {
                "detect_interval": 1
            }
        }
    },
    "74": {
        "camera_name": "74",
        "angle": 180,
        "url": "rtsp://admin:wspw123456@192.168.0.10/Streaming/Channels/1",
        "models": {
            "phone": {
                "detect_interval": 4
            },
            "smoking": {
                "detect_interval": 2
            },
            "face": {
                "detect_interval": 4,
                "face_feature_config": {
                    "face_path": "/home/vision/Desktop/face/\u767d\u540d\u53551.csv",
                    "type": "white"
                }
            },
            "safetyhat": {
                "detect_interval": 1
            }
        }
    }
}