import cv2 as cv

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv.VideoCapture("Videos/Face-Cap-3.mp4")
FourCC = cv.VideoWriter_fourcc(*"XVID")
result = cv.VideoWriter("selfproject1.avi", FourCC, 20.0, (640, 480))


while cap.isOpened():
    _, frame = cap.read()
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_detect = face_cascade.detectMultiScale(gray_img, 1.1, 4)

    for (x, y, w, h) in face_detect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=3)

        cv.imshow("Frame", frame)
        result.write(frame)

    if cv.waitKey(1) & 0xFF == ord("e"):
        break

cap.release()
result.release()
cv.destroyAllWindows()