
#Import necessary libraries
from flask import Flask, render_template, Response, request
import torch
import cv2
import numpy as np
from pymongo import MongoClient
from PIL import Image
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    set_logging
from utils.torch_utils import select_device, time_synchronized
import base64
from numpy import random
#Initialize the Flask app
import base64

app = Flask(__name__)

def return_img_stream(img_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def predicta(frame):

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # Set Dataloader & Run inference
    img = letterbox(frame, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    # pred = model(img, augment=opt.augment)[0]
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres)
    #
    # # Process detections
    ret = []
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]}'
                prob = round(float(conf) * 100, 2)  # round 2
                ret_i = [label, prob, xyxy]
                ret.append(ret_i)
    # return
    # label  'face' 'smoke' 'drink' 'phone'
    # prob
    # xyxy position
    if ret == []:
        print("detect no driver!")
        return "no driver"
    else:
        # print("prediction:", ret)
        print("prediction result:", ret[0][0])
        return ret[0][0]


def fetch_img():
    query_img = pictures.find({}, {"_id": 0, "timestamp": 1, "images": 1, "size": 1}).sort("timestamp", -1).limit(1)
    tmp_img = query_img[0]
    img = tmp_img["images"]
    img1 = Image.frombytes("RGB", size=tmp_img["size"], data=img)
    img1 = np.array(img1)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = img1[:, :, ::-1]  # BGR -> RGB

    ret, buffer = cv2.imencode('.jpg', img1)
    frame = buffer.tobytes()
    img_stream = base64.b64encode(frame).decode()

    return img_stream

def fetch_stream():
    while (True):
        query_img = pictures.find({}, {"_id": 0, "timestamp": 1, "images": 1, "size": 1}).sort("timestamp", -1).limit(1)
        tmp_img = query_img[0]
        img = tmp_img["images"]
        img1 = Image.frombytes("RGB", size=tmp_img["size"], data=img)
        img1 = np.array(img1)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # img1 = img1[:, :, ::-1]  # BGR -> RGB

        ret, buffer = cv2.imencode('.jpg', img1)
        frame = buffer.tobytes()
        # frame = base64.b64encode(frame1).decode()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/')
def hello_world():
    img_path = './testflask.jpg'
    img_stream = return_img_stream(img_path)
    # img_stream = fetch_img()
    return render_template('index.html',
                           img_stream=img_stream)
        # index1 是逐帧

@app.route('/', methods=['POST','GET'])  # for: index1 逐帧
def update_pic():
    if request.method == "POST":
        if request.form["submit_button"] == "Manual mode":
            img_s = fetch_img()
            return render_template('index1.html', img_stream=img_s, situation = 1)


@app.route('/video_feed')  # bytes， index
def video_feed():
    # render_template()
    return Response(fetch_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


print("Connect to mongoDB")
mongodbUri = 'mongodb://user2:myadmin@101.35.252.209:27017/admin'
# user1，myadmin
# user2，myadmin
# user3，myadmin
client = MongoClient(mongodbUri)
mydb = client['images']
pictures = mydb["pictures"]
respo = mydb["responses"]
print("Successfully connect to mongoDB")

imgsz = 640
weights = r'weights/best.pt'
opt_device = ''
opt_conf_thres = 0.6
opt_iou_thres = 0.45
device = select_device(opt_device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
if half:
    model.half()  # to FP16
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

app.run(host="10.0.12.11",port=5000)


