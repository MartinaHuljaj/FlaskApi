
from fileinput import filename
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import getData
from keras.layers import StringLookup
from flask import Flask, jsonify,request
import cv2
import CTCLoss
from PIL import Image
import split_into_letters
import firebase_admin
from firebase_admin import credentials,storage
print(firebase_admin)
cred = credentials.Certificate("./key.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'scriptit-ed51c.appspot.com'})
bucket=storage.bucket()


app = Flask(__name__)
@app.route('/api',methods=['POST','GET'])

def model():
    request_data=request.data
    request_data=json.loads(request_data.decode('utf-8'))
    file_name=request_data['fileN']
    user=request_data['user']
    print(user)
    print(file_name)

    blob=bucket.get_blob(user + "/images/" + file_name)
    print(list(bucket.list_blobs()))
    print(blob)
    arr=np.frombuffer(blob.download_as_string(), np.uint8)
    print(arr)
    img=cv2.imdecode(arr,cv2.COLOR_GRAY2BGR)

    data=split_into_letters.data_preparation(img)
    new_model=tf.keras.models.load_model(r'C:\Users\Martina\Documents\GitHub\FlaskApi\char_model')
    new_model.summary()
    prediction= new_model.predict(data)
    print(prediction)

    # characters=[]
    # character_file=open(r"C:\Users\mhuljaj\Documents\TestiranjeOcr\data\class.txt","r").readlines()

    # #izbacivanje komentara i rijeci koje mozda imaju gresku
    # for line in character_file:
    #     characters.append(line[:len(line)-1])
        

    # char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
    # num_to_char = StringLookup(
    #     vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    # )

    # # A utility function to decode the output of the network.
    # def decode_batch_predictions(pred):
    #     input_len = np.ones(pred.shape[0]) * pred.shape[1]
    #     # Use greedy search. For complex tasks, you can use beam search.
    #     results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
    #         :, :21
    #     ]
    #     # Iterate over the results and get back the text.
    #     output_text = []
    #     for res in results:
    #         res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
    #         res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
    #         output_text.append(res)
    #     return output_text
    
    
    return jsonify({'fileN': prediction})