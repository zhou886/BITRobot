import vision
import virtualImage
import time
from multiprocessing import Process, Queue

from yolov5.utils.general import LOGGER


def main():
    q1 = Queue()  # 视觉模块发给虚拟形象模块的信号队列
    q2 = Queue()  # 语音模块发给虚拟形象模块的信号队列
    visionProc = Process(target=vision.videoCap, args=(q1,))
    # testProc = Process(target = read, args=(q1,))
    virtualImageProc = Process(target=virtualImage.show, args=(q1, q2, ))
    visionProc.start()
    # testProc.start()
    visionProc.join()
    # testProc.terminate()
    pass


def read(q):
    while True:
        value = q.get(True)
        print('Get %s from queue' % value)
        time.sleep(0.1)


if __name__ == '__main__':
    main()
