import cv2
import numpy as np
import random
import os
import shutil
import argparse


# 图片亮度变换
def brightness_aug(img_dir, save_img_dir):
    for img_file in os.listdir(img_dir):
        fname = img_file.split('.jpg')[0]
        source_path = os.path.join(img_dir, img_file)
        img = cv2.imread(source_path)

        parameter = random.choice(list(range(30, 80)) + list(range(-80, -30)))

        # 增加图像亮度
        res = np.uint8(np.clip((1 * np.int16(img) + parameter), 0, 255))
        img_save_path = os.path.join(save_img_dir, fname + '.png')
        cv2.imwrite(img_save_path, res)  # 随机改变图片亮度的新图片存入图片文件夹


# 图片对比度变换
def contrast_aug(img_dir, save_img_dir):
    for img_file in os.listdir(img_dir):
        fname = img_file.split('.jpg')[0]
        source_path = os.path.join(img_dir, img_file)

        parameter = random.uniform(1, 3)
        parameter = round(parameter, 1)

        img = cv2.imread(source_path)
        res = np.uint8(np.clip((parameter * (np.int16(img) - 60) + 50), 0, 255))

        img_save_path = os.path.join(save_img_dir, fname + '.png')
        cv2.imwrite(img_save_path, res)  # 随机改变图片亮度的新图片存入图片文件夹


# 图片模糊度变换
def blur_aug(img_dir, save_img_dir):
    for img_file in os.listdir(img_dir):
        fname = img_file.split('.jpg')[0]
        source_path = os.path.join(img_dir, img_file)

        kernel_size = (5, 5)
        sigma = random.uniform(2, 5)
        sigma = round(sigma, 1)

        img = cv2.imread(source_path)
        res = cv2.GaussianBlur(img, kernel_size, sigma)

        img_save_path = os.path.join(save_img_dir, fname + '.png')
        cv2.imwrite(img_save_path, res)  # 随机改变图片模糊度的新图片存入图片文件夹


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='../datasets/coco1000/images/val2017', help='img path')
    parser.add_argument('--label_dir', type=str, default='../datasets/coco1000/labels/val2017', help='label path')
    parser.add_argument('--save_dir', type=str, default='../datasets/obj_aug_coco1000', help='save path')
    arg = parser.parse_args()

    img_dir, label_dir, save_dir = arg.img_dir, arg.label_dir, arg.save_dir
    brightness_save_img_dir = os.path.join(save_dir, 'brightness/images')
    brightness_save_label_dir = os.path.join(save_dir, 'brightness/labels')
    contrast_save_img_dir = os.path.join(save_dir, 'contrast/images')
    contrast_save_label_dir = os.path.join(save_dir, 'contrast/labels')
    blur_save_img_dir = os.path.join(save_dir, 'blur/images')
    blur_save_label_dir = os.path.join(save_dir, 'blur/labels')

    os.makedirs(brightness_save_img_dir, exist_ok=True)
    os.makedirs(contrast_save_img_dir, exist_ok=True)
    os.makedirs(blur_save_img_dir, exist_ok=True)

    brightness_aug(img_dir, brightness_save_img_dir)
    contrast_aug(img_dir, contrast_save_img_dir)
    blur_aug(img_dir, blur_save_img_dir)

    shutil.copytree(label_dir, brightness_save_label_dir)
    shutil.copytree(label_dir, contrast_save_label_dir)
    shutil.copytree(label_dir, blur_save_label_dir)
