import cv2 as cv

def videoCap():
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Do not find your camera.")
            break
        cv.imshow("frame", frame)
        if cv.waitKey(50) & 0xff == ord('q'):
            break

def main():
    videoCap()

if __name__ == '__main__':
    main()