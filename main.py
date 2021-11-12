import vision
from multiprocessing import Process, Queue

def main():
    q = Queue()
    visionProc = Process(target = vision.videoCap, args=(q,))
    #testProc = Process(target = read, args=(q,))
    visionProc.start()
    #testProc.start()
    visionProc.join()
    #testProc.terminate()
    pass

def read(q):
    while True:
        value = q.get(True)
        print('Get %s from queue' % value)

if __name__ == '__main__':
    main()