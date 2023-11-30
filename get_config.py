import requests
import json

def get_model_config(model_url):
    response = requests.get(model_url)
    python_code=f"model_config={json.dumps(response.json(),indent=4)}"
    print(python_code)
    with open('model_config.py',"w") as py_file:
        py_file.write(python_code)

def get_camera_config(url):
    response = requests.get(url)
    print(response.json())
    python_code=f"camera_config={json.dumps(response.json(),indent=4)}"
    with open('camera_config.py',"w") as py_file:
        py_file.write(python_code)
