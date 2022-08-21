import numpy as np
import tensorflow as tf
from tensorflow import keras
import getData
from keras.layers import StringLookup

def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

data=getData.get_data("C:\\Users\\mhuljaj\\Documents\\TestiranjeOcr\\lines.png")
new_model=tf.keras.models.load_model(r'C:\Users\mhuljaj\Documents\TestiranjeOcr\my_test_model', custom_objects={"CTCLoss": CTCLoss})
new_model.summary()
prediction= new_model.predict(data)
print(prediction)

characters=[]
character_file=open(r"C:\Users\mhuljaj\Documents\TestiranjeOcr\data\class.txt","r").readlines()

#izbacivanje komentara i rijeci koje mozda imaju gresku
for line in character_file:
  characters.append(line[:len(line)-1])
    

char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

# A utility function to decode the output of the network.
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :21
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

print(decode_batch_predictions(prediction))