import os
import face_recognition
import cv2
import numpy as np
import json
import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin import firestore
from google.cloud import storage
from urllib.request import urlopen
import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define DIR
DIR_GOOGLE_CLOUD_CREDENTIAL = os.getenv("DIR_GOOGLE_CLOUD_CREDENTIAL")
DIR_BUCKET_STORAGE = os.getenv("DIR_BUCKET_STORAGE")
FACE_ENCODED_FILE_NAME = os.getenv("FACE_ENCODED_FILE_NAME")

cred = credentials.Certificate(f"{DIR_GOOGLE_CLOUD_CREDENTIAL}")
firebase_admin.initialize_app(cred)

db = firestore.client()
bucket_name = f'{DIR_BUCKET_STORAGE}'
database = {}


def generate_image_url(bucket_name, blob_path):
    """ generate signed URL of a image stored on google storage. 
        Valid for 300 seconds in this case. You can increase this 
        time as per your requirement. 
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)                                                        
    blob = bucket.blob(blob_path) 
    return blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')

def encode_from_gcs(bucket_name, prefix, delimiter=None):
    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
    lastEncoded = np.zeros(128)
    imgFile = []

    # Note: The call returns a response only when the iterator is consumed.
    for blob in blobs:
        x = blob.name.split("/")

        if ".jpg" not in x[-1] and x[1] != '':
            pass
        else:
            try:
                blob_name = blob.name
                url = generate_image_url(bucket_name, blob_name)
                s = urlopen(url).read()
                imgFile.append(x[-1])
                img = np.asarray(bytearray(s), dtype="uint8")
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faceLoc = face_recognition.face_locations(img)
                img_enc = face_recognition.face_encodings(img, faceLoc)[0]
                temp = lastEncoded + img_enc
                lastEncoded = temp
            except IndexError: 
                pass
    return lastEncoded, len(imgFile)


def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    """Lists all the blobs in the bucket that begin with the prefix.

    This can be used to list all blobs in a "folder", e.g. "public/".

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:

        a/1.txt
        a/b/2.txt

    If you specify prefix ='a/', without a delimiter, you'll get back:

        a/1.txt
        a/b/2.txt

    However, if you specify prefix='a/' and delimiter='/', you'll get back
    only the file directly under 'a/':

        a/1.txt

    As part of the response, you'll also get back a blobs.prefixes entity
    that lists the "subfolders" under `a/`:

        a/b/
    """

    storage_client = storage.Client()
    name = []

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    # Note: The call returns a response only when the iterator is consumed.
    # print("Blobs:")
    for blob in blobs:
        # print(blob.name)
        pass

    if delimiter:
        # print("Prefixes:")
        for prefix in blobs.prefixes:
            x = prefix.split("/")
            name.append(x[1])
    return name


name = list_blobs_with_prefix(bucket_name, prefix='imagesAttendance/', delimiter='/')
print('Process Image Encoding... ')

for i in name:
    imgFile = []
    total, lenFile = encode_from_gcs(bucket_name, prefix='imagesAttendance/'+i)
    avg = total/lenFile
    database[i] = avg
print("Encoding Berhasil")
print('Process membungkus Image Encoding ke dalam file .json...')

#upload pickle/json file .pkl/.json to google cloud storage in firebase console
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)      
blob = bucket.blob(f"{FACE_ENCODED_FILE_NAME}")
blob.upload_from_string(data=json.dumps(database, cls=NumpyEncoder),content_type='application/json')  
print("Upload file .json ke Google Cloud Storage via Firebase berhasil")


