import datetime

import numpy as np
import cv2
import torch
from numpy import random
from pymongo import MongoClient
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    set_logging
from utils.torch_utils import select_device, time_synchronized
import base64
from PIL import Image
import time
# 10.201.81.203
#    子网掩码  . . . . . . . . . . . . : 255.255.224.0

class camera_g:
    def __init__(self):

        self.window_name = "camera-server"

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
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

    def predicta(self, frame):
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        # Set Dataloader & Run inference
        img = self.letterbox(frame, new_shape=imgsz)[0]

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

    def upload_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        size = (frame.shape[1],frame.shape[0])
        img_bytes = frame.tobytes()
        # img_bytes = base64.b64encode(img_bytes).decode("utf-8")

        # thetime = datetime.datetime.utcnow()
        # timestr = datetime.datetime.strftime(thetime, '%d:%m:%Y:%H:%M:%S')
        #
        # pic_doc = {"timestamp": timestr, "images": img_bytes, "size": size}
        # pictures.insert_one(pic_doc)

        print(len(img_bytes))

    def upload_response(self, response):

        thetime = datetime.datetime.utcnow()
        timestr = datetime.datetime.strftime(thetime, '%d:%m:%Y:%H:%M:%S')
        # doc["response"] = response
        # doc["timestamp"] = timestr
        # doc["images"] = ""
        res_doc = {"timestamp": timestr, "response": response}
        respo.insert_one(res_doc)
        res_time = datetime.datetime.now()
        print("upload a response: ", (res_time-thetime).microseconds)



    def capture_image(self, cap):
        cv2.namedWindow(self.window_name)  # camera's name
        # cap = cv2.VideoCapture(camera_idx)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow(self.window_name, frame)
            c = cv2.waitKey(1)

            if c & 0xFF == ord('q'):
                break
            if c == ord("n"):
                print(frame.shape)
                # self.upload_image(frame)
                res = self.predicta(frame)
                self.upload_response(res)
            # self.update_fps()

    def get_image(self):
        while(True):
            print("==========================")
            thetime = datetime.datetime.now()
            query_img = pictures.find({}, {"_id":0, "timestamp":1, "images": 1, "size":1}).sort("timestamp", -1).limit(1)
            try:
                tmp_img = query_img[0]
            except:
                print("find no picture")
                time.sleep(2)
                continue

            img = tmp_img["images"]
            img1 = Image.frombytes("RGB", size=tmp_img["size"], data=img)
            process_time = datetime.datetime.now()
            print("query and process done, time: ", (process_time - thetime).microseconds)

            img1 = np.array(img1)
            cv2.imshow(self.window_name, img1)
            c = cv2.waitKey(1)

            resp = self.predicta(img1)
            predict_time = datetime.datetime.now()
            print("predict done, since processing: ", (predict_time-process_time).microseconds)

            if (( thetime -(datetime.datetime.strptime(tmp_img["timestamp"], '%d:%m:%Y:%H:%M:%S')) ).seconds < 5):
                self.upload_response(resp)
                fin_time = datetime.datetime.now()
                print("finish query + predict + response, since the loop: ", (fin_time-thetime).microseconds)
            else:
                print("too previous, no response")
                continue

    def run2(self):
        self.get_image()

    def run(self):
        cap = cv2.VideoCapture(0)
        self.capture_image(cap)
        cap.release()
        cv2.destroyWindow(self.window_name)


if __name__ == '__main__':
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

    print(names)
    a = camera_g()
    a.run2()
    # a.run()