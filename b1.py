
from hashlib import sha512
from pymongo import MongoClient
from flask import Flask, render_template, Response, request,flash, redirect
from PIL import Image # Pillow library
from base64 import b64decode
import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    set_logging
from utils.torch_utils import select_device, time_synchronized
import base64
from numpy import random
import datetime
import requests
import json


app = Flask('__name__')

USER_OK = 0
USER_NOT_EXIST = 1
USER_BAD_PASSWORD = 2

def finduser(userid):
    # Check if "userid" exists in the database. If so, return the user record, otherwise
    # return None

    # We use "find_one" to locate the user since we should only have one record. Recall we
    # earlier retrieved the "users" collection from our database.

    return users.find_one({"userid": userid})


def check_pw(userid, passwd):
    # Check if the user exists
    user_data = finduser(userid)

    if user_data is None:
        # User does not exist.
        return USER_NOT_EXIST

    # User exists, check password by hashing the provided password
    # and comparing with the one that is stored.
    pw = sha512(passwd.encode('utf-8')).hexdigest()

    if pw == user_data["password"]:
        return USER_OK
    else:
        return USER_BAD_PASSWORD
# Great, let's now create our / endpoint.


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
        # print("detect no driver!")
        return "no driver"
    else:
        # print("prediction:", ret)
        # print("prediction result:", ret[0][0])
        return ret[0][0]

def upload_actuators(resp):
    thetime = datetime.datetime.utcnow()
    timestr = datetime.datetime.strftime(thetime, '%d:%m:%Y:%H:%M:%S')
    res_doc = {"timestamp": timestr, "response": resp}
    respo.insert_one(res_doc)
    res_time = datetime.datetime.now()
    print("Server upload a response: ", (res_time - thetime).microseconds)

def detect_get_response(image):
    timea = datetime.datetime.now()
    resul = predicta(image)
    timef = datetime.datetime.now()
    print(resul, " is the result of prediction, consume: ", (timef-timea).microseconds)
    return resul, 200


#     return "Dandelion", 200
@app.route('/')
def root():
    # Let's see if a name is provided
    # name = request.args.get('name')
    #
    # # Return back a text message and a result code of 200, which means success.
    # if name is not None:
    #     return "Hello " + name, 200
    # else:
    #     return "Hello whoever you are.", 200
    return render_template("index2.html", situation = "no driver")

@app.route('/detect', methods=['POST'])
def detect_image():
    try:
        # Get the JSON data
        data = request.get_json()

        # Check that userid, password and image are all present
        # if 'userid' not in data or 'password' not in data or 'image' not in data:
        #     return "Must provide userid, password and image data.", 400
        #
        # # Check the password
        # time1 = datetime.datetime.now()
        # res = check_pw(data['userid'], data['password'])
        # time2 = datetime.datetime.now()
        # print("check the user and password done, consume: ", (time2 - time1).microseconds)
        #
        #
        # if res == USER_NOT_EXIST:
        #     return "User does not exist.", 400
        #
        #
        #
        # if res == USER_BAD_PASSWORD:
        #     return "Incorrect password", 400

        # if check_pics(data['image']) == PICTURE_NOT_EXIST:
        #     pictures.insert_one({"picture_base64": data['image']})

            # User ID and password match. Let's extract the image data

        # img_data = b64decode(data['images'])  # if the client encode

        time3 = datetime.datetime.now()
        img_data = data["images"]
        img_data = b64decode(img_data)  # if the client encode
        img = Image.frombytes("RGB", size=data["size"], data=img_data)
        img1 = np.array(img)
        time4 = datetime.datetime.now()
        print("Server preprocesses the image, consume:", (time4-time3).microseconds)


        ret, buffer = cv2.imencode('.jpg', img1)
        frame = buffer.tobytes()
        img_stream = base64.b64encode(frame).decode()
        time5 = datetime.datetime.now()
        print("Server process image stream for web done, consume:", (time5-time4).microseconds)
        # img = Image.frombytes("RGB", (249, 249), img_data)  # picture

        # Display the image. Comment this part out for Question 1
        #         img.show()

        # Return the classification
        res = detect_get_response(img1)
        upload_actuators(res)
        render_template("index2.html", img_stream=img_stream, situation=res)
        # return render_template("index2.html", img_stream=img_stream, situation = res)
        return res
        # return "OK", 200
    except Exception as e:
        # Return error message
        return "Error processing request." + str(e), 500


print("Connect to mongoDB")
mongodbUri = 'mongodb://user2:myadmin@101.35.252.209:27017/admin'
# user1，myadmin
# user2，myadmin
# user3，myadmin
client = MongoClient(mongodbUri)
mydb = client['images']
pictures = mydb["pictures"]
respo = mydb["responses"]
users = mydb["users"]
print("Successfully connect to mongoDB")


imgsz = 640
weights = r'weights/best.pt'
opt_device = ''
opt_conf_thres = 0.3
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

app.run(debug=True, host="10.0.12.11")
