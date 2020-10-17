import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='images'
images=[]
classnames=[]
mylist=os.listdir(path)
print(mylist)

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])

print(classnames)

# Encoding process
def findencoding(images):
    encodelist=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode =face_recognition.face_encodings(img)[0]
        encodelist.append(encode)

    return encodelist

def markAttendance(name):
    with open('attandance.csv','r+') as f:
        mydatalist = f.readlines()
        nameList = []
        for line in mydatalist:
            entry = line.split(',')
            nameList.append(entry[0])

            now=datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.seek(0)
            f.writelines(f'\n{name},{dtstring}')
            f.truncate()



encodelistknow =findencoding(images)
print('encoding complete')


# using webcame
cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgsmall = cv2.resize(img,(0,0),None,0.25,0.25)#reduce the image size
    imgsmall = cv2.cvtColor(imgsmall,cv2.COLOR_BGR2RGB)

    facecurrframe = face_recognition.face_locations(imgsmall)
    encodecurrframe = face_recognition.face_encodings(imgsmall,facecurrframe)

    for encodeface,faceloc in zip(encodecurrframe,facecurrframe):
        matches=face_recognition.compare_faces(encodelistknow,encodeface)
        facedis= face_recognition.face_distance(encodelistknow,encodeface)
        print(facedis)
        matchindex=np.argmin(facedis)


        if matches[matchindex]:
            name=classnames[matchindex].upper()
            print(name)
            y1,x2,y2,x1=faceloc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_ITALIC,1,(255,0,255),2)
            markAttendance(name)

    cv2.imshow('webame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()