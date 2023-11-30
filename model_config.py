model_config={
    "safetyhat": {
        "weights": "./Model/safety_hemlt.pt",
        "img_size": 640,
        "conf_thres": 0.5,
        "event": "safetyhat"
    },
    "face": {
        "weights": "./Model/face_best.pt",
        "img_size": 640,
        "conf_thres": 0.5,
        "event": "face"
    },
    "phone": {
        "weights": "./Model/phone_2023_11_25.pt",
        "img_size": 416,
        "conf_thres": 0.7,
        "event": "phone"
    },
    "smoking": {
        "weights": "./Model/smoking-v1-2023-11-25.pt",
        "img_size": 416,
        "conf_thres": 0.5,
        "event": "smoking"
    }
}