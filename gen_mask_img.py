# 读取coco图片，选定一个框进行mask，然后保存对应的 mask png图片，以及对应新的label
import cv2
import numpy as np
import os
import argparse


coco_clean_classes = {0, 2, 4, 6, 8, 10, 11, 15, 16, 22, 23, 46, 48, 49, 53, 54, 61, 72, 74}
openimage_clean_classes = {110, 121, 228, 34, 404, 426, 433, 440, 445, 455, 467, 483, 498, 503, 510, 570, 592, 595, 68, 98}
voc_clean_classes = {0, 1, 2, 4, 5, 6, 7, 11, 14, 18}


# (w1,h1)是原始图片的尺寸，x包含 object label的位置信息
def mask(img, x, img_save_path, mask_save_path):
    cv2.imwrite(img_save_path, img)
    cv2.destroyAllWindows()

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
    print("左上x坐标:{}".format(top_left_x))
    print("左上y坐标:{}".format(top_left_y))
    print("右下x坐标:{}".format(bottom_right_x))
    print("右下y坐标:{}".format(bottom_right_y))

    black = np.zeros((h1, w1))  # 全黑底色图片，与原图的尺寸一样大
    white_color = (255, 255, 255)
    cv2.rectangle(black, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)),
                  color=white_color, thickness=-1)
    # cv2.imshow('show', black)
    cv2.imwrite(mask_save_path, black)
    # cv2.waitKey(0)  # 按键结束
    cv2.destroyAllWindows()


def gen_mask(image_path, label_path, img_save_path, mask_save_path, random_mask, mask_id):
    # 读取图像文件
    img = cv2.imread(str(image_path))
    # 读取 label
    with open(label_path, 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

    if random_mask:
        area_li = []
        for i in range(len(lb)):
            x = lb[i]
            label, x, y, w, h = x
            area_li.append(w * h)
        mask_id = area_li.index(min(area_li))
        x = lb[mask_id]  # 第一个 object label对应的位置信息
        mask(img, x, img_save_path, mask_save_path)
    else:
        x = lb[mask_id]  # 第一个 object label对应的位置信息
        mask(img, x, img_save_path, mask_save_path)


# 存储每张图片的 mask，以及增加 mask之后的 label txt文件
def get_coco(img_dir, label_dir, save_img_tmp_dir, save_img_dir, save_label_dir):
    for img_file in os.listdir(img_dir):
        fname = img_file.split('.jpg')[0]
        if os.path.exists(os.path.join(label_dir, fname + '.txt')):  # 找到匹配的 jpg 和 txt
            jpg_path = os.path.join(img_dir, img_file)
            txt_path = os.path.join(label_dir, fname + '.txt')
            f = open(txt_path)
            lines = f.readlines()  # 原始 label
            # 保存 mask png图片
            mask_save_path = os.path.join(save_img_tmp_dir, fname + '_mask.png')
            img_save_path = os.path.join(save_img_tmp_dir, fname + '.png')

            if len(lines) > 1:  # label 框多于1个的时候才做删除，不然就变成没有label的图片了
                gen_mask(jpg_path, txt_path, img_save_path, mask_save_path, random_mask=True, mask_id=-1)
                with open(os.path.join(save_label_dir, fname + '_mask.txt'), "w") as f:  # 保存新的 label txt
                    f.writelines(lines[1:])
            else:  # label 框只有1个的时候，直接把原始图片和label保存到对应的文件夹
                img_save_path = os.path.join(save_img_dir, fname + '.png')
                img = cv2.imread(str(jpg_path))
                cv2.imwrite(img_save_path, img)  # 保存原始的图片
                cv2.destroyAllWindows()
                with open(os.path.join(save_label_dir, fname + '.txt'), "w") as f:  # 保存 label txt
                    f.writelines(lines)


# 存储每张图片的 mask，以及增加 mask之后的 label txt文件
def get_specific_coco(img_dir, label_dir, save_img_tmp_dir, save_img_dir, save_label_tmp_dir, save_label_dir, dataset):
    for img_file in os.listdir(img_dir):
        fname = img_file.split('.jpg')[0]
        if os.path.exists(os.path.join(label_dir, fname + '.txt')):  # 找到匹配的 jpg 和 txt
            jpg_path = os.path.join(img_dir, img_file)
            txt_path = os.path.join(label_dir, fname + '.txt')
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            obj_class = [int(lines[i].split(' ')[0]) for i in range(len(lines))]  # 每个label 对应的 class
            print(obj_class)
            del_ids = []
            if dataset == 'coco':
                del_ids = [idx for (idx, tmp) in enumerate(obj_class) if tmp in coco_clean_classes]
            elif dataset == 'openimage':
                del_ids = [idx for (idx, tmp) in enumerate(obj_class) if tmp in openimage_clean_classes]
            elif dataset == 'voc':
                del_ids = [idx for (idx, tmp) in enumerate(obj_class) if tmp in voc_clean_classes]
            # 保存 mask png图片
            mask_save_path = os.path.join(save_img_tmp_dir, fname + '_mask.png')
            img_save_path = os.path.join(save_img_tmp_dir, fname + '.png')

            delete = True
            if len(del_ids) > 0:
                del_id = del_ids[0]

                with open(txt_path, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
                label, x, y, w, h = lb[del_id]
                area = w * h
                if area <= 0.5:  # 删除的面积占比小于0.5才行
                    gen_mask(jpg_path, txt_path, img_save_path, mask_save_path, random_mask=False, mask_id=del_id)
                    with open(os.path.join(save_label_tmp_dir, fname + '_mask.txt'), "w") as f:  # 保存新的 label txt
                        f.writelines(lines[0:del_id] + lines[del_id+1:])
                    with open(os.path.join(save_label_tmp_dir, fname + '_replace.txt'), "w") as f:  # 保存待替换的label txt
                        f.writelines(lines[del_id])
                else:
                    delete = False
            if delete == False or len(del_ids) == 0:  # 查找失败，没有可以替换的图片，直接把原始图片和label保存到对应的文件夹
                img_save_path = os.path.join(save_img_dir, fname + '.png')
                img = cv2.imread(str(jpg_path))
                cv2.imwrite(img_save_path, img)  # 保存原始的图片
                cv2.destroyAllWindows()
                with open(os.path.join(save_label_dir, fname + '.txt'), "w") as f:  # 保存 label txt
                    f.writelines(lines)


def prepare_remove(dataset, type):
    if dataset == 'coco' and type == 'val':
        img_dir = "../datasets/coco1000/images/val2017"
        label_dir = "../datasets/coco1000/labels/val2017"
        save_img_tmp_dir = "../datasets/obj_aug_coco1000/remove/images_tmp"
        save_img_dir = "../datasets/obj_aug_coco1000/remove/images"
        save_label_dir = "../datasets/obj_aug_coco1000/remove/labels"

    elif dataset == 'coco' and type == 'train':
        img_dir = "../datasets/coco1000/images/train2017"
        label_dir = "../datasets/coco1000/labels/train2017"
        save_img_tmp_dir = "../datasets/coco1000/images/train2017_aug/remove/images_tmp"
        save_img_dir = "../datasets/coco1000/images/train2017_aug/remove"
        save_label_dir = "../datasets/coco1000/labels/train2017_aug/remove"

    elif dataset == 'openimage' and type == 'val':
        img_dir = "E:/code/mmdetection/data/OpenImages1000/OpenImages/validation"
        label_dir = "E:/code/mmdetection/data/OpenImages1000/labels"
        save_img_tmp_dir = "../datasets/obj_aug_openimages1000/remove/images_tmp"
        save_img_dir = "../datasets/obj_aug_openimages1000/remove/images"
        save_label_dir = "../datasets/obj_aug_openimages1000/remove/labels"

    elif dataset == 'voc' and type == 'val':
        img_dir = "E:/code/mmdetection/data/VOCdevkit/test/VOC2007/JPEGImages1000"
        label_dir = "E:/code/mmdetection/data/VOCdevkit/test/labels"
        save_img_tmp_dir = "../datasets/obj_aug_voc1000/remove/images_tmp"
        save_img_dir = "../datasets/obj_aug_voc1000/remove/images"
        save_label_dir = "../datasets/obj_aug_voc1000/remove/labels"

    os.makedirs(save_img_tmp_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    get_coco(img_dir, label_dir, save_img_tmp_dir, save_img_dir, save_label_dir)  # 接下来，save_img_dir可以直接做为 lama 的输入文件夹，lama的输出文件夹与save_label_dir组合起来，就可以用于yolo检测验证了


def prepare_replace(dataset, type):
    if dataset == 'coco' and type == 'val':
        img_dir = "../datasets/coco1000/images/val2017"
        label_dir = "../datasets/coco1000/labels/val2017"
        save_img_tmp_dir = "../datasets/obj_aug_coco1000/replace/images_tmp1"
        save_img_dir = "../datasets/obj_aug_coco1000/replace/images"
        save_label_tmp_dir = "../datasets/obj_aug_coco1000/replace/labels_tmp"
        save_label_dir = "../datasets/obj_aug_coco1000/replace/labels"

    elif dataset == 'coco' and type == 'train':
        img_dir = "../datasets/coco1000/images/train2017"
        label_dir = "../datasets/coco1000/labels/train2017"
        save_img_tmp_dir = "../datasets/coco1000/images/train2017_aug/replace/images_tmp1"
        save_img_dir = "../datasets/coco1000/images/train2017_aug/replace"
        save_label_tmp_dir = "../datasets/coco1000/labels/train2017_aug/replace_tmp"
        save_label_dir = "../datasets/coco1000/labels/train2017_aug/replace"

    elif dataset == 'openimage' and type == 'val':
        img_dir = "E:/code/mmdetection/data/OpenImages1000/OpenImages/validation"
        label_dir = "E:/code/mmdetection/data/OpenImages1000/labels"
        save_img_tmp_dir = "../datasets/obj_aug_openimages1000/replace/images_tmp1"
        save_img_dir = "../datasets/obj_aug_openimages1000/replace/images"
        save_label_tmp_dir = "../datasets/obj_aug_openimages1000/replace/labels_tmp"
        save_label_dir = "../datasets/obj_aug_openimages1000/replace/labels"

    elif dataset == 'voc' and type == 'val':
        img_dir = "E:/code/mmdetection/data/VOCdevkit/test/VOC2007/JPEGImages1000"
        label_dir = "E:/code/mmdetection/data/VOCdevkit/test/labels"
        save_img_tmp_dir = "../datasets/obj_aug_voc1000/replace/images_tmp1"
        save_img_dir = "../datasets/obj_aug_voc1000/replace/images"
        save_label_tmp_dir = "../datasets/obj_aug_voc1000/replace/labels_tmp"
        save_label_dir = "../datasets/obj_aug_voc1000/replace/labels"

    os.makedirs(save_img_tmp_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_label_tmp_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    get_specific_coco(img_dir, label_dir, save_img_tmp_dir, save_img_dir, save_label_tmp_dir, save_label_dir, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='val', required=True, choices=['remove', 'replace'])
    parser.add_argument('--dataset', required=True, choices=['coco', 'voc', 'openimage'])
    parser.add_argument('--type', default='val', required=True, choices=['val', 'train'])
    args = parser.parse_args()

    if args.task == 'remove':
        prepare_remove(args.dataset, args.type)
    if args.task == 'replace':
        prepare_replace(args.dataset, args.type)
