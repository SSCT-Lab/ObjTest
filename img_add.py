import cv2
import numpy as np
import math
import random
import os
import argparse
from class_info import coco_class_dict, openimages_class_dict, voc_class_dict


def add_alpha_channel(img):
    """ 为jpg图像添加alpha通道 """

    b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道

    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
    return img_new


def add_mask(source_path, mask_path, mask_class, resize_rate=0.05, random_pos=True, pos_h=None, pos_w=None):
    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    source_img = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)

    h, w = mask_img.shape[0], mask_img.shape[1]
    H, W = source_img.shape[0], source_img.shape[1]
    k = w/h

    if source_img.ndim == 2:  # 灰度图片先转成三通道的图片
        source_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2RGB)

    # 判断jpg图像是否已经为4通道
    if source_img.shape[2] == 3:
        source_img = add_alpha_channel(source_img)

    # 面积比 (w * (w/k))/ (W*H) <= resize_rate
    resize_w = int(math.sqrt(resize_rate * H * W * k)) if int(math.sqrt(resize_rate * H * W * k)) >= 1 else 1
    resize_h = int(resize_w/k) if int(resize_w/k) >= 1 else 1  # 不能变成 0了，至少要是1
    resized_mask_img = cv2.resize(mask_img, dsize=(resize_w, resize_h))
    if random_pos:  # pos是mask的左上角坐标，只要长宽不超出原图的边界即可
        pos_h = random.randint(0, H-resize_h)
        pos_w = random.randint(0, W-resize_w)
    else:  # 固定 mask 的位置，需要检查是否超出边界
        assert pos_w is not None
        assert pos_h is not None
        resize_w = W - pos_w if (pos_w + resize_w > W) else resize_w
        resize_h = H - pos_h if (pos_h + resize_h > H) else resize_h

    # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
    alpha_png = resized_mask_img[0:resize_h, 0:resize_w, 3] / 255.0
    alpha_jpg = 1 - alpha_png
    # 开始叠加
    for c in range(0, 3):
        source_img[pos_h:(pos_h + resize_h), pos_w:(pos_w + resize_w), c] = (
                    (alpha_jpg * source_img[pos_h:(pos_h + resize_h), pos_w:(pos_w + resize_w), c]) + (alpha_png * resized_mask_img[0:resize_h, 0:resize_w, c]))
    # print("resize mask w:", resize_w, "resize mask h:", resize_h, "\npos w:", pos_w, "pos h:", pos_h)
    # print("source w:", W, "source h:", H)
    yolo_label_info = yolo_label(pos_w, pos_h, pos_w + resize_w, pos_h + resize_h, H, W, mask_class)
    return source_img, yolo_label_info


def yolo_label(x1, y1, x2, y2, H, W, mask_class):
    new_x = (x1 + x2) / (2 * W)
    new_y = (y1 + y2) / (2 * H)

    new_w = (x2 - x1) / W
    new_h = (y2 - y1) / H

    # yolo_label_info = str(mask_class) + ' ' + new_x + ' ' + new_y + ' ' + new_w + ' ' + new_h + '\n'
    yolo_label_info = '{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(str(mask_class), new_x, new_y, new_w, new_h)
    return yolo_label_info


def get_mask_pos(image_path, x):
    img = cv2.imread(str(image_path))
    h1, w1 = img.shape[:2]
    label, x, y, w, h = x
    print("原图宽高:\nw1={}\nh1={}".format(w1, h1))

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

    mask_area = (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
    size_ratio = mask_area / (w1 * h1)
    print("size_ratio:", size_ratio)

    return int(top_left_x), int(top_left_y), size_ratio


def gen_mask(img_dir, label_dir, mask_dir, save_img_dir, save_label_dir):  # insertion
    for img_file in os.listdir(img_dir):
        fname = img_file.split('.jpg')[0]
        print(fname)
        if os.path.exists(os.path.join(label_dir, fname + '.txt')):  # 找到匹配的 jpg 和 txt
            source_path = os.path.join(img_dir, img_file)
            txt_path = os.path.join(label_dir, fname + '.txt')
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            random_mask = random.sample(os.listdir(mask_dir), 1)
            mask_path = os.path.join(mask_dir, random_mask[0])
            mask_class = random_mask[0].split('_')[0]

            added_mask_img, mask_yolo_label = add_mask(source_path, mask_path, mask_class)
            lines.append(mask_yolo_label)  # 在原始 label的基础上把 mask的 label加上
            img_save_path = os.path.join(save_img_dir, fname + '.png')
            cv2.imwrite(img_save_path, added_mask_img)  # 增加了mask的图片存入图片文件夹
            with open(os.path.join(save_label_dir, fname + '.txt'), "w") as f:  # 增加了 mask的新 label存入txt文件夹
                f.writelines(lines)


def replace_mark(img_dir, label_dir, mask_dir, save_img_dir, save_label_dir, dataset):
    for img_file in os.listdir(img_dir):
        fname = img_file.split('_mask.png')[0]
        print(fname)
        if os.path.exists(os.path.join(label_dir, fname + '_mask.txt')):  # 找到匹配的 jpg 和 txt
            source_path = os.path.join(img_dir, img_file)
            ori_txt_path = os.path.join(label_dir, fname + '_mask.txt')
            replace_txt_path = os.path.join(label_dir, fname + '_replace.txt')
            with open(ori_txt_path, 'r') as f1:
                lines = f1.readlines()
            with open(replace_txt_path, 'r') as f2:
                line = f2.readline()
                lb = [float(i) for i in line.strip().split(' ')]
            replace_class = int(lb[0])
            if dataset == 'coco':
                mask_path = os.path.join(mask_dir, coco_class_dict[replace_class])
            elif dataset == 'openimage':
                mask_path = os.path.join(mask_dir, openimages_class_dict[replace_class])
            elif dataset == 'voc':
                mask_path = os.path.join(mask_dir, voc_class_dict[replace_class])
            pos_w, pos_h, size_ratio = get_mask_pos(source_path, lb)

            added_mask_img, mask_yolo_label = add_mask(source_path, mask_path, replace_class, resize_rate=size_ratio,
                                                       random_pos=False, pos_h=pos_h, pos_w=pos_w)
            lines.append(mask_yolo_label)  # 在原始 label的基础上把 mask的 label加上
            img_save_path = os.path.join(save_img_dir, fname + '.png')
            cv2.imwrite(img_save_path, added_mask_img)  # 增加了mask的图片存入图片文件夹
            with open(os.path.join(save_label_dir, fname + '.txt'), "w") as f:  # 增加了 mask的新 label存入txt文件夹
                f.writelines(lines)


def insertion(dataset, type):
    if dataset == 'coco' and type == 'val':
        mask_dir = "./coco_classes"
        img_dir = "../datasets/coco1000/images/val2017"
        label_dir = "../datasets/coco1000/labels/val2017"
        save_img_dir = "../datasets/obj_aug_coco1000/insertion/images"
        save_label_dir = "../datasets/obj_aug_coco1000/insertion/labels"

    elif dataset == 'coco' and type == 'train':
        mask_dir = "./coco_classes"
        img_dir = "../datasets/coco1000/images/train2017"
        label_dir = "../datasets/coco1000/labels/train2017"
        save_img_dir = "../datasets/coco1000/images/train2017_aug/insertion"
        save_label_dir = "../datasets/coco1000/labels/train2017_aug/insertion"

    elif dataset == 'openimage' and type == 'val':
        mask_dir = "./openimages_classes"
        img_dir = "E:/code/mmdetection/data/OpenImages1000/OpenImages/validation"
        label_dir = "E:/code/mmdetection/data/OpenImages1000/labels"
        save_img_dir = "../datasets/obj_aug_openimages1000/insertion/images"
        save_label_dir = "../datasets/obj_aug_openimages1000/insertion/labels"

    elif dataset == 'voc' and type == 'val':
        mask_dir = "./voc_classes"
        img_dir = "E:/code/mmdetection/data/VOCdevkit/test/VOC2007/JPEGImages1000"
        label_dir = "E:/code/mmdetection/data/VOCdevkit/test/labels"
        save_img_dir = "../datasets/obj_aug_voc1000/insertion/images"
        save_label_dir = "../datasets/obj_aug_voc1000/insertion/labels"

    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    gen_mask(img_dir, label_dir, mask_dir, save_img_dir, save_label_dir)


def replace(dataset, type):
    if dataset == 'coco' and type == 'val':
        mask_dir = "./coco_classes"
        img_dir = "../datasets/coco1000/images/train2017_aug/replace/images_tmp2"
        label_dir = "../datasets/coco1000/labels/train2017_aug/replace_tmp"
        save_img_dir = "../datasets/coco1000/images/train2017_aug/replace"
        save_label_dir = "../datasets/coco1000/labels/train2017_aug/replace"
    elif dataset == 'coco' and type == 'train':
        mask_dir = "./coco_classes"
        img_dir = "../datasets/obj_aug_coco1000/replace/images_tmp2"
        label_dir = "../datasets/obj_aug_coco1000/replace/labels_tmp"
        save_img_dir = "../datasets/obj_aug_coco1000/replace/images"
        save_label_dir = "../datasets/obj_aug_coco1000/replace/labels"
    elif dataset == 'openimage' and type == 'val':
        mask_dir = "./openimages_classes"
        img_dir = "../datasets/obj_aug_openimages1000/replace/images_tmp2"
        label_dir = "../datasets/obj_aug_openimages1000/replace/labels_tmp"
        save_img_dir = "../datasets/obj_aug_openimages1000/replace/images"
        save_label_dir = "../datasets/obj_aug_openimages1000/replace/labels"
    elif dataset == 'voc' and type == 'val':
        mask_dir = "./voc_classes"
        img_dir = "../datasets/obj_aug_voc1000/replace/images_tmp2"
        label_dir = "../datasets/obj_aug_voc1000/replace/labels_tmp"
        save_img_dir = "../datasets/obj_aug_voc1000/replace/images"
        save_label_dir = "../datasets/obj_aug_voc1000/replace/labels"

    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    replace_mark(img_dir, label_dir, mask_dir, save_img_dir, save_label_dir, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=['insertion', 'replace'])
    parser.add_argument('--dataset', required=True, choices=['coco', 'voc', 'openimage'])
    parser.add_argument('--type', default='val', required=True, choices=['val', 'train'])
    args = parser.parse_args()

    if args.task == 'insertion':
        insertion(args.dataset, args.type)
    if args.task == 'replace':
        replace(args.dataset, args.type)
