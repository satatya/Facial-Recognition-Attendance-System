import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from tkinter import Tk, Label, Button, filedialog, messagebox

path = 'ImagesAttendance'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

def startRecognition():
    encodeListKnown = findEncodings(images)
    print('Encoding complete')

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceSCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceSCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, faceSCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def selectFolder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        global path, images, classNames, mylist
        path = folder_selected
        images = []
        classNames = []
        mylist = os.listdir(path)
        print(mylist)
        for cl in mylist:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
        print(classNames)
        messagebox.showinfo("Info", "Images loaded successfully")

root = Tk()
root.title("Face Recognition Attendance System")

label = Label(root, text="Face Recognition Attendance System", font=("Helvetica", 16))
label.pack(pady=20)

load_button = Button(root, text="Load Images Folder", command=selectFolder)
load_button.pack(pady=10)

start_button = Button(root, text="Start Recognition", command=startRecognition)
start_button.pack(pady=10)

quit_button = Button(root, text="Quit", command=root.quit)
quit_button.pack(pady=10)

root.mainloop()
