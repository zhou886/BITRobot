from math import frexp
import cv2 as cv
from multiprocessing import Queue

import torch
from MyNetwork import MyNetwork
import time
from torchvision.transforms import *


def videoCap(q) -> None:
    cap = cv.VideoCapture(0)
    state = False
    record = []
    point = 0
    leng = 10
    myNetwork = MyNetwork()
    myNetwork.load_state_dict(torch.load('test.pth', map_location=torch.device('cpu')))
    for i in range(leng):
        record.append(False)
    trans = Compose([
        ToTensor(),
        Resize([64, 64])
    ])
    face_detect = cv.CascadeClassifier(
        r'C:\Users\ZYX\AppData\Local\Programs\Python\Python38\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can not find your camera.")
            break
        cv.imshow("frame", frame)
        img = trans(frame)
        img = torch.unsqueeze(img, dim=0)
        # face = face_detect.detectMultiScale(frame)
        res = myNetwork(img)
        # if len(face) > 0:
        #     res = True
        # else:
        #     res = False
        predict, index = torch.max(res, 1)
        record[point] = index[0]
        print(index[0])
        point = (point+1) % 10
        if judge(record, point) == 1 and not state:
            state = True
            q.put("True")
            print("True")
        if judge(record, point) == -1 and state:
            state = False
            q.put("False")
            print("False")

        # time.sleep(0.5)
        if cv.waitKey(100) & 0xff == ord('q'):
            break


def judge(record, point, tar=5) -> int:
    tot = 0
    for i in range(tar):
        tot += record[(point-i+len(record)) % len(record)]
    if (tot == tar):
        return 1
    if (tot == 0):
        return -1
    return 0


if __name__ == '__main__':
    q = Queue()
    videoCap(q)
