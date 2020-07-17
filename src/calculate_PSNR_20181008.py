
# coding: utf-8


import os
import math
import numpy as np
from PIL import Image
from scipy import misc
import imageio

def convert_rgb_to_y(image):
    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]])
    y_image = image.dot(xform.T) + 16.0

    return y_image
def convert_rgb_to_ycbcr(image):
    if len(image.shape) < 2 or image.shape[2] == 1:
        return image

    xform = np.array(
        [[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0],
        [- 37.945 / 256.0, - 74.494 / 256.0, 112.439 / 256.0],
        [112.439 / 256.0, - 94.154 / 256.0, - 18.285 / 256.0]])

    ycbcr_image = image.dot(xform.T)
    ycbcr_image[:, :, 0] += 16.0
    ycbcr_image[:, :, [1, 2]] += 128.0

    return ycbcr_image
def compute_mse(image1, image2, border_size=0):
    """
    Computes MSE from 2 images.
    We round it and clip to 0 - 255. Then shave it from 6 + scale.
    """
    if len(image1.shape) == 2:
        image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
    if len(image2.shape) == 2:
        image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)

    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
        return None

    image1 = trim_image_as_file(image1)
    image2 = trim_image_as_file(image2)

    diff = np.subtract(image1, image2)
    if border_size > 0:
        diff = diff[border_size:-border_size, border_size:-border_size, :]
    mse = np.mean(np.square(diff))

    return mse
def trim_image_as_file(image):
    image = np.round(image)
    return np.clip(image, 0, 255)
def get_psnr(mse, max_value=255.0):
    if mse is None or mse == float('Inf') or mse == 0:
        psnr = 0
    else:
        psnr = 20 * math.log(max_value / math.sqrt(mse), 10)
    return psnr
def load_image(filename, width=0, height=0, channels=0, alignment=0, print_console=True):
    image = misc.imread(filename)
    return image
'''
    if len(image.shape) == 2:
        image = image.reshape(image.shape[0], image.shape[1], 1)
    if (width != 0 and image.shape[1] != width) or (height != 0 and image.shape[0] != height):
        raise LoadError("Attributes mismatch")
    if channels != 0 and image.shape[2] != channels:
        raise LoadError("Attributes mismatch")
    if alignment != 0 and ((width % alignment) != 0 or (height % alignment) != 0):
        raise LoadError("Attributes mismatch")

    # if there is alpha plane, cut it
    if image.shape[2] >= 4:
        image = image[:, :, 0:3]

    if print_console:
        print("Loaded [%s]: %d x %d x %d" % (filename, image.shape[1], image.shape[0], image.shape[2]))
'''
    
def set_image_alignment(image, alignment):
    alignment = int(alignment)
    width, height = image.shape[1], image.shape[0]
    width = (width // alignment) * alignment
    height = (height // alignment) * alignment

    if image.shape[1] != width or image.shape[0] != height:
        image = image[:height, :width, :]

    if len(image.shape) >= 3 and image.shape[2] >= 4:
        image = image[:, :, 0:3]

    return image
def build_input_image(image, width=0, height=0, channels=1, scale=1, alignment=0, convert_ycbcr=True, jpeg_mode=False):
    """
    build input image from file.
    crop, adjust the image alignment for the scale factor, resize, convert color space.
    """

    if width != 0 and height != 0:
        if image.shape[0] != height or image.shape[1] != width:
            x = (image.shape[1] - width) // 2
            y = (image.shape[0] - height) // 2
            image = image[y: y + height, x: x + width, :]

    if image.shape[2] >= 4:
        image = image[:, :, 0:3]

    if alignment > 1:
        image = set_image_alignment(image, alignment)

    if scale != 1:
        image = resize_image_by_pil(image, 1.0 / scale)

    if channels == 1 and image.shape[2] == 3:
        if convert_ycbcr:
            image = convert_rgb_to_y(image, jpeg_mode=jpeg_mode)
    else:
        if convert_ycbcr:
            image = convert_rgb_to_ycbcr(image, jpeg_mode=jpeg_mode)

    return image
def get_psnr(mse, max_value=255.0):
    if mse is None or mse == float('Inf') or mse == 0:
        psnr = 0
    else:
        psnr = 20 * math.log(max_value / math.sqrt(mse), 10)
    return psnr
def do_for_evaluate(file_path1,file_path2):
    print_console=False
    true_image = set_image_alignment(load_image(file_path1, print_console=False), 2)
    
    true_y_image = convert_rgb_to_y(true_image)
    input_bicubic_y_image = convert_rgb_to_y(set_image_alignment(load_image(file_path2, print_console=False), 2))
    mse = compute_mse(true_y_image, input_bicubic_y_image, border_size=2)

    return mse

def get_memary_img(gr,sr):
    true_y_image = convert_rgb_to_y(gr)
    input_bicubic_y_image = convert_rgb_to_y(sr)
    mse = compute_mse(true_y_image, input_bicubic_y_image, border_size=2)

    return mse

def get_file_psnr(gr_dir='',sr_dir='',scale=2):

    psnrlist=np.empty((1,0))
    evetest =np.zeros((1,1))
    img_files = os.listdir(sr_dir)
    for img in img_files:
        input_sr = imageio.imread(sr_dir+"/"+img)
        name=img.split('x')[0]+'.'+img.split('.')[-1]
        try:
            gr_img=imageio.imread(gr_dir+"/"+name)
        except:
            print('not found {}!'.format(name))

        evetest[0] = get_psnr(get_memary_img(input_sr,gr_img))
        psnrlist=np.append(psnrlist,evetest,axis=1)
    print(psnrlist)
    print(np.mean(psnrlist, axis=1))


get_file_psnr(gr_dir='/home/yx/桌面/Newcode/benchmark/Set5/HR',
            sr_dir='/home/yx/桌面/Newcode/experiment/edsr_x4/results-Demo',scale=4)

 
    
