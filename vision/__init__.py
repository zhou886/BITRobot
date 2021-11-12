import cv2 as cv
from multiprocessing import Queue
from yolov5.detect import run

def videoCap(q):
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Do not find your camera.")
            break
        cv.imshow("frame", frame)
        cv.imwrite('./yolov5/data/images/src1.jpg', frame)
        flag = run()
        if flag and q.empty():
            # q.put('DETECTED')
            print("DETECTED!")
        if cv.waitKey(1000) & 0xff == ord('q'):
            break

if __name__ == '__main__':
    q = Queue()
    videoCap(q)