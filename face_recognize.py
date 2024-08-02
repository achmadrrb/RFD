import os
import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin import firestore
from datetime import datetime, timedelta
import pytz
import cv2
import numpy as np
import face_recognition
from getmac import get_mac_address as gma
import json
from google.cloud.firestore_v1.base_query import FieldFilter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define DIR
DIR_GOOGLE_CLOUD_CREDENTIAL = os.getenv("DIR_GOOGLE_CLOUD_CREDENTIAL")
DIR_BUCKET_STORAGE = os.getenv("DIR_BUCKET_STORAGE")
FACE_ENCODED_FILE_NAME = os.getenv("FACE_ENCODED_FILE_NAME")

cred = credentials.Certificate(f"{DIR_GOOGLE_CLOUD_CREDENTIAL}")
app = firebase_admin.initialize_app(cred)
db = firestore.client()

doc_presensi = db.collection(u'presensi')
doc_rekap = db.collection(u'rekap')

def markAttendance(List):
    tz_JKT = pytz.timezone('Asia/Jakarta')
    datetime_JKT = datetime.now(tz_JKT)
    faces = np.array(List, dtype='object')
    for i, face in enumerate(faces):
        if len(face) > 6:
            nameList = []
            name = face[0]
            
            # Menyimpan nama di database
            docs_akun = db.collection("akun").where(filter=FieldFilter("nama", "==", name)).stream()
            doc_stream = db.collection(u'presensi').stream()
            for doc_a in docs_akun:
                    job = doc_a.to_dict().get('jenis_pekerjaan')
                    akun_id = doc_a.to_dict().get('id')
            for doc in doc_stream:
                nameList.append(doc.to_dict().get("nama"))
            if akun_id != "":
                if name not in nameList:
                    doc_presensi.document(akun_id).set({
                        u'nama': name,
                        u'akun_id': akun_id,
                        u'alat_id': gma(),
                        u'datetime': datetime_JKT,
                        u'jenis_pekerjaan': job
                    })
                    doc_rekap.add({
                        'nama': name,
                        'akun_id': akun_id,
                        'alat_id': gma(),
                        'datetime': datetime_JKT,
                        'jenis_pekerjaan': job
                    })

            if name in nameList:
                docs = db.collection("rekap").where(filter=FieldFilter("nama", "==", name)).stream()
                same_date = []

                for doc in docs:
                    date_rec = doc.to_dict().get('datetime')
                    diff = datetime_JKT - date_rec
                    # Jika teredeteksi pada hari yang sama, maka data diperbarui
                    if date_rec.date() == datetime_JKT.date():
                        id_doc = doc.id
                        same_date.append(date_rec.date())
                if len(same_date) == 0:
                    doc_presensi.document(akun_id).set({
                        u'nama': name,
                        u'akun_id': akun_id,
                        u'alat_id': gma(),
                        u'datetime': datetime_JKT,
                        u'jenis_pekerjaan': job
                    })
                    doc_rekap.add({
                        'nama': name,
                        'akun_id': akun_id,
                        'alat_id': gma(),
                        'datetime': datetime_JKT,
                        'jenis_pekerjaan': job
                    })

                elif same_date[0] == datetime_JKT.date():        
                    if diff > timedelta(minutes=5):
                        doc_presensi.document(akun_id).set({
                            u'nama': name,
                            u'akun_id': akun_id,
                            u'alat_id': gma(),
                            u'datetime': datetime_JKT,
                            u'jenis_pekerjaan': job
                        })
                        
                        doc_rekap.document(id_doc).set({
                            u'nama': name,
                            u'akun_id': akun_id,
                            u'alat_id': gma(),
                            u'datetime': datetime_JKT,
                            u'jenis_pekerjaan': job
                        })



#retrieving json from firebase
bucket = storage.bucket(f'{DIR_BUCKET_STORAGE}')
blob = bucket.blob(f"{FACE_ENCODED_FILE_NAME}") 
downloaded_json= json.loads(blob.download_as_text(encoding="utf-8"))
classNames = []
encodeListKnown = []
for key, value in downloaded_json.items():
    classNames.append(key)
    encodeListKnown.append(value)

cap = cv2.VideoCapture(0)
new_width, new_height = 1920, 1080

# Atur resolusi baru untuk kamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Resolusi Kamera: {}x{}".format(int(width), int(height)))


categories = {} 
category_index = None 
face_in_frame = 0
frame_count = 0

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for i, (encodeFace, faceLoc) in enumerate(zip(encodesCurFrame, facesCurFrame)):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.4)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        try:
            if matches[matchIndex]:
                name = classNames[matchIndex]

                # Cari indeks kategori berdasarkan nama wajah
                category_index = next((index for index, cat in enumerate(categories) if cat[0] == name), None)
                if category_index is None:
                    # Jika kategori belum ada, buat list baru untuk wajah baru
                    new_category = [name]
                    # Tambahkan list wajah baru ke categories
                    categories.append(new_category)
                else:
                    # Jika kategori sudah ada, tambahkan wajah ke kategori yang sudah ada
                    categories[category_index].append(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

            else:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "Unknown", (x1+6, y2-6),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        except IndexError:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, 'Unkwown', (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    if facesCurFrame:
        face_in_frame += 1

    if face_in_frame == 10:
        markAttendance(categories)
        categories.clear()
        face_in_frame = 0
    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(33)==27:    # Esc key to stop
        break
