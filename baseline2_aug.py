import cv2
import numpy as np
import random
import os
import argparse


def get_coord(x, y, w, h, w1, h1):
    # 边界框反归一化
    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1

    # 计算坐标
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y


def get_yolo_label(x1, y1, x2, y2, H, W, mask_class):
    new_x = (x1 + x2) / (2 * W)
    new_y = (y1 + y2) / (2 * H)

    new_w = (x2 - x1) / W
    new_h = (y2 - y1) / H

    # yolo_label_info = str(mask_class) + ' ' + new_x + ' ' + new_y + ' ' + new_w + ' ' + new_h + '\n'
    yolo_label_info = '{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(str(mask_class), new_x, new_y, new_w, new_h)
    return yolo_label_info


def get_crop_label(img, max_width, max_hight, label_path):
    # 读取 label
    with open(label_path, 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

    h1, w1 = img.shape[:2]
    save_label_li = []

    for i in range(len(lb)):
        label_x = lb[i]
        label, x, y, w, h = label_x

        top_left_x, top_left_y, bottom_right_x, bottom_right_y = get_coord(x, y, w, h, w1, h1)
        if bottom_right_x < max_width and bottom_right_y < max_hight:  #原始边框完全在新的图片范围内
            tmp = get_yolo_label(top_left_x, top_left_y, bottom_right_x, bottom_right_y, max_hight, max_width, int(label))
            save_label_li.append(tmp)

    return save_label_li


# 随机图片裁剪，删除算子的暴力版本
def crop(img_dir, label_dir, save_img_dir, save_label_dir):
    for img_file in os.listdir(img_dir):
        fname = img_file.split('.jpg')[0]
        if os.path.exists(os.path.join(label_dir, fname + '.txt')):  # 找到匹配的 jpg 和 txt
            jpg_path = os.path.join(img_dir, img_file)
            txt_path = os.path.join(label_dir, fname + '.txt')
            img = cv2.imread(str(jpg_path))

            hight, width = img.shape[0], img.shape[1]
            random_width = random.randint(1, width)
            random_hight = random.randint(1, hight)
            cropped = img[0:random_hight, 0:random_width]
            img_save_path = os.path.join(save_img_dir, fname + '.png')
            cv2.imwrite(img_save_path, cropped)  # 新图片存入图片文件夹

            save_label_li = get_crop_label(img, random_width, random_hight, txt_path)
            with open(os.path.join(save_label_dir, fname + '.txt'), "w") as f:  # 保存新的 label txt
                f.writelines(save_label_li)


def integration_label(img1, img2, label_path1, label_path2, H, W):  # 1在左，2在右
    new_label_li = []
    h1, w1 = img1.shape[:2]
    with open(label_path1, 'r') as f:
        lb1 = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
    for i in range(len(lb1)):
        label_x = lb1[i]
        label, x, y, w, h = label_x
        top_left_x1, top_left_y1, bottom_right_x1, bottom_right_y1 = get_coord(x, y, w, h, w1, h1)
        tmp = get_yolo_label(top_left_x1, top_left_y1, bottom_right_x1, bottom_right_y1, H, W, int(label))
        new_label_li.append(tmp)

    h2, w2 = img2.shape[:2]
    with open(label_path2, 'r') as f:
        lb2 = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
    for j in range(len(lb2)):
        label_x = lb2[j]
        label, x, y, w, h = label_x
        top_left_x2, top_left_y2, bottom_right_x2, bottom_right_y2 = get_coord(x, y, w, h, w2, h2)
        tmp = get_yolo_label(top_left_x2 + w1, top_left_y2, bottom_right_x2 + w1, bottom_right_y2, H, W, int(label))
        new_label_li.append(tmp)
    return new_label_li


# 随机图片拼贴， 增加算子的暴力版本
def integration(img_dir, label_dir, save_img_dir, save_label_dir):
    img_dir_li = os.listdir(img_dir)
    for i in range(len(img_dir_li)-1):
        img_file1, img_file2 = img_dir_li[i], img_dir_li[i+1]
        source_path1, source_path2 = os.path.join(img_dir, img_file1), os.path.join(img_dir, img_file2)
        img1, img2 = cv2.imread(source_path1), cv2.imread(source_path2)
        fname1, fname2 = img_file1.split('.jpg')[0], img_file2.split('.jpg')[0]

        if os.path.exists(os.path.join(label_dir, fname1 + '.txt')) and os.path.exists(os.path.join(label_dir, fname2 + '.txt')):  # 找到匹配的 jpg 和 txt
            txt_path1 = os.path.join(label_dir, fname1 + '.txt')
            txt_path2 = os.path.join(label_dir, fname2 + '.txt')

            img1_h, img2_h = img1.shape[0], img2.shape[0]
            img1_w, img2_w = img1.shape[1], img2.shape[1]
            W = img1_w + img2_w

            if img1_h > img2_h:
                img2_resize = cv2.copyMakeBorder(img2, 0, img1_h - img2_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                res = cv2.hconcat([img1, img2_resize])  # 水平拼接
                new_label_li = integration_label(img1, img2, txt_path1, txt_path2, img1_h, W)
            elif img2_h > img1_h:
                img1_resize = cv2.copyMakeBorder(img1, 0, img2_h - img1_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                res = cv2.hconcat([img2, img1_resize])  # 水平拼接
                new_label_li = integration_label(img2, img1, txt_path2, txt_path1, img2_h, W)
            else:
                res = cv2.hconcat([img1, img2])  # 水平拼接
                new_label_li = integration_label(img1, img2, txt_path1, txt_path2, img1_h, W)

            img_save_path = os.path.join(save_img_dir, fname1 + '.png')
            cv2.imwrite(img_save_path, res)  # 新图片存入图片文件夹

            with open(os.path.join(save_label_dir, fname1 + '.txt'), "w") as f:  # 保存新的 label txt
                f.writelines(new_label_li)


def add_mask(image_path, label_path, img_save_path, mask_id):
    # 读取图像文件
    img = cv2.imread(str(image_path))
    # 读取 label
    with open(label_path, 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
    label_x = lb[mask_id]  # 第 mask_id个 object label对应的位置信息
    label, x, y, w, h = label_x
    h1, w1 = img.shape[:2]
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = get_coord(x, y, w, h, w1, h1)
    black_color = (0, 0, 0)
    cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)),
                  color=black_color, thickness=-1)
    cv2.imwrite(img_save_path, img)


# 选中一个目标物体直接贴黑块，替换算子的暴力版本
def mask(img_dir, label_dir, save_img_dir, save_label_dir):
    for img_file in os.listdir(img_dir):
        fname = img_file.split('.jpg')[0]
        if os.path.exists(os.path.join(label_dir, fname + '.txt')):  # 找到匹配的 jpg 和 txt
            jpg_path = os.path.join(img_dir, img_file)
            txt_path = os.path.join(label_dir, fname + '.txt')
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            img_save_path = os.path.join(save_img_dir, fname + '.png')
            if len(lines) > 1:  # label 框多于1个的时候才做删除，不然就变成没有label的图片了
                add_mask(jpg_path, txt_path, img_save_path, mask_id=0)
                with open(os.path.join(save_label_dir, fname + '_mask.txt'), "w") as f:  # 保存新的 label txt
                    f.writelines(lines[1:])
            else:  # 查找失败，没有可以替换的图片，直接把原始图片和label保存到对应的文件夹
                img_save_path = os.path.join(save_img_dir, fname + '.png')
                img = cv2.imread(str(jpg_path))
                cv2.imwrite(img_save_path, img)  # 保存原始的图片
                with open(os.path.join(save_label_dir, fname + '.txt'), "w") as f:  # 保存 label txt
                    f.writelines(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='../datasets/coco1000/images/val2017', help='img path')
    parser.add_argument('--label_dir', type=str, default='../datasets/coco1000/labels/val2017', help='label path')
    parser.add_argument('--save_dir', type=str, default='../datasets/obj_aug_coco1000', help='save path')
    arg = parser.parse_args()

    img_dir, label_dir, save_dir = arg.img_dir, arg.label_dir, arg.save_dir
    crop_save_img_dir = os.path.join(save_dir, 'crop/images')
    crop_save_label_dir = os.path.join(save_dir, 'crop/labels')
    integration_save_img_dir = os.path.join(save_dir, 'integration/images')
    integration_save_label_dir = os.path.join(save_dir, 'integration/labels')
    mask_save_img_dir = os.path.join(save_dir, 'mask/images')
    mask_save_label_dir = os.path.join(save_dir, 'mask/labels')

    os.makedirs(crop_save_img_dir, exist_ok=True)
    os.makedirs(crop_save_label_dir, exist_ok=True)
    os.makedirs(integration_save_img_dir, exist_ok=True)
    os.makedirs(integration_save_label_dir, exist_ok=True)
    os.makedirs(mask_save_img_dir, exist_ok=True)
    os.makedirs(mask_save_label_dir, exist_ok=True)

    crop(img_dir, label_dir, crop_save_img_dir, crop_save_label_dir)
    integration(img_dir, label_dir, integration_save_img_dir, integration_save_label_dir)
    mask(img_dir, label_dir, mask_save_img_dir, mask_save_label_dir)
