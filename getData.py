import splitingTheLine
import numpy as np
from flask import Flask

app = Flask(__name__)
@app.route('/api')

def get_data(img, file_name):
    image_parts=splitingTheLine.splittingTheLine(img, file_name)
    print(image_parts)
    final_image_parts=splitingTheLine.saving_images(image_parts)
    data=np.array(final_image_parts)
    return data
