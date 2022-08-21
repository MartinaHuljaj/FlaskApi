import cv2
import numpy as np
from PIL import Image
import image_slicer
from image_slicer import slice
import os.path
import tensorflow as tf
import pickle


def splittingTheLine(image_path):
    
    #read the image of lines
    img = Image.open(image_path)
    file_name=os.path.basename(image_path).split('.')[0]
    print(file_name)
    #get the size of image
    img_w,img_h=img.size
    print(img_w)
    print(img_h)


    #image height to 64 pixels
    wpercent = (64/float(img.size[1]))
    wsize = int((float(img.size[0])*float(wpercent)))
    img = img.resize((wsize,64), Image.Resampling.LANCZOS)
    img=img.convert('RGB')
    print(img.size)
    img_w=img.size[0]

    #slice the image into 256x64 images
    print(img_w//256)
    padding_left_right=int((1-(img_w/256-img_w//256))*256)

    print(padding_left_right)
    if(padding_left_right%256==0):
        padding_left_right=0

    if(padding_left_right==0):
        number_of_images=int(img_w//256 )
    else:
        number_of_images=int(img_w//256 + 1)
    new_width=padding_left_right+img_w
    print(number_of_images)
    padding_img = Image.new(mode="RGB",size= (img_w+padding_left_right, 64),color= (255, 255, 255))
    padding_img.paste(img, (padding_left_right, 0))
    padding_img.save(image_path)
    print(padding_img.size)

    if number_of_images>1:
        parts_of_image=slice(image_path, col=number_of_images, row=1)
    print(parts_of_image[0])
    image_names=[]
    for image in parts_of_image:
        image_names.append(image_slicer.main.Tile.generate_filename(image,directory=os.path.dirname(image_path), prefix=file_name, format='png', path=True))
    print(image_names)
    return image_names


def saving_images(image_paths):
    images=[]
    for path in image_paths:
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, 1)
        images.append(image)
    images=np.array(images)
    images=images.astype('float32') / 255.0
    return images

def get_data(path):
    image_parts=splittingTheLine(path)
    print(image_parts)
    final_image_parts=saving_images(image_parts)
    data=np.array(final_image_parts)
    return data

get_data("C:\\Users\\mhuljaj\\Documents\\TestiranjeOcr\\lines.png")
