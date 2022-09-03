import cv2
from imutils import contours
import numpy as np
from PIL import Image

def split_into_letters(image):
# Load image, grayscale, Otsu's threshold
    #image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
    # Find contours, sort from left-to-right, then crop
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method="left-to-right")
    images=[]
    # Filter using contour area and extract ROI
    ROI_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 10:
            x,y,w,h = cv2.boundingRect(c)
            ROI = image[y:y+h, x:x+w]
            image_name=str(ROI_number)+'.png'
            cv2.imwrite('{}.png' .format(ROI_number),  ROI)
            ROI_number += 1
            images.append(image_name)
    
    return images


def images_resize(letters):
    images=[]
    for img_path in letters:
        img=Image.open(img_path)
        img_w,img_h=img.size
        print(img_w)
        print(img_h)
        if(img_h>img_w):
            wpercent = (28/float(img.size[1]))
            wsize = int((float(img.size[0])*float(wpercent)))
            img = img.resize((wsize,28), Image.Resampling.LANCZOS)
        else:
            wpercent = (28/float(img.size[0]))
            wsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((28,wsize), Image.Resampling.LANCZOS)
        padding_img = Image.new(mode="RGB",size= (28,28),color= (255, 255, 255))
        padding_img.paste(img)
        padding_img.save(img_path)
        print(padding_img.size)
        padding_img=np.array(padding_img)
        images.append(padding_img)

    images=np.array(images)
    images=images.astype('float32') / 255.0
    return images

def data_preparation(original_image_path):
    letters=split_into_letters(original_image_path)
    data=images_resize(letters)
    return data


data_preparation('test4.png')


