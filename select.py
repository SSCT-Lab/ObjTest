import os
import random
import shutil
import numpy as np


def random_sample(img_in_path, label_in_path, img_out_path, label_out_path, m=100):
    source_paths, txt_paths, fnames = [], [], []
    for img_file in os.listdir(img_in_path):
        fname = img_file.split('.')[0]
        if os.path.exists(os.path.join(label_in_path, fname + '.txt')):  # 找到匹配的 jpg 和 txt
            source_paths.append(os.path.join(img_in_path, img_file))
            txt_paths.append(os.path.join(label_in_path, fname + '.txt'))
            fnames.append(fname)
    idx = list(range(0, len(source_paths)))
    selected_idx = random.sample(idx, m)
    for i in selected_idx:
        shutil.copyfile(source_paths[i], os.path.join(img_out_path, fnames[i] + '.jpg'))
        shutil.copyfile(txt_paths[i], os.path.join(label_out_path, fnames[i] + '.txt'))


def get_aug(insertion_dir, remove_dir, replace_dir, out_dir):
    out_img_dir = os.path.join(out_dir, 'images')
    out_label_dir = os.path.join(out_dir, 'labels')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    insertion_img = os.path.join(insertion_dir, 'images')
    insertion_label = os.path.join(insertion_dir, 'labels')
    for img_file in os.listdir(insertion_img):
        fname = img_file.split('.png')[0]
        img_path = os.path.join(insertion_img, img_file)
        shutil.copyfile(img_path, os.path.join(out_img_dir, fname + '_insert.png'))
    for txt_file in os.listdir(insertion_label):
        fname = txt_file.split('.txt')[0]
        txt_path = os.path.join(insertion_label, txt_file)
        shutil.copyfile(txt_path, os.path.join(out_label_dir, fname + '_insert.txt'))

    remove_img = os.path.join(remove_dir, 'images')
    remove_label = os.path.join(remove_dir, 'labels')
    for img_file in os.listdir(remove_img):
        fname = img_file.split('.png')[0]
        img_path = os.path.join(remove_img, img_file)
        shutil.copyfile(img_path, os.path.join(out_img_dir, fname + '_remove.png'))
    for txt_file in os.listdir(remove_label):
        fname = txt_file.split('.txt')[0]
        txt_path = os.path.join(remove_label, txt_file)
        shutil.copyfile(txt_path, os.path.join(out_label_dir, fname + '_remove.txt'))

    replace_img = os.path.join(replace_dir, 'images')
    replace_label = os.path.join(replace_dir, 'labels')
    for img_file in os.listdir(replace_img):
        fname = img_file.split('.png')[0]
        img_path = os.path.join(replace_img, img_file)
        shutil.copyfile(img_path, os.path.join(out_img_dir, fname + '_replace.png'))
    for txt_file in os.listdir(replace_label):
        fname = txt_file.split('.txt')[0]
        txt_path = os.path.join(replace_label, txt_file)
        shutil.copyfile(txt_path, os.path.join(out_label_dir, fname + '_replace.txt'))


def get_select(path_file, label_in_path, img_out_path, label_out_path, m=100):
    file_li = np.load(path_file)
    print(file_li[0])
    source_paths, txt_paths, fnames = [], [], []
    for i in range(len(file_li)):
        source_path = file_li[i]
        fname = source_path.split('\\')[-1].split('.')[0]
        if os.path.exists(os.path.join(label_in_path, fname + '.txt')):  # 找到匹配的 jpg 和 txt
            source_paths.append(source_path)
            txt_paths.append(os.path.join(label_in_path, fname + '.txt'))
            fnames.append(fname)
    print(len(source_paths), len(fnames))
    for i in range(m):
        shutil.copyfile(source_paths[i], os.path.join(img_out_path, fnames[i] + '.jpg'))
        shutil.copyfile(txt_paths[i], os.path.join(label_out_path, fnames[i] + '.txt'))


if __name__ == '__main__':
    # # 从 Tsample 中随机选择 100个用例
    # img_sample_path = 'E:/code/datasets/coco1000/images/val2017'
    # label_sample_path = 'E:/code/datasets/coco1000/labels/val2017'
    # img_sample100_path = 'E:/code/datasets/obj_aug_coco1000/select/sample/images'
    # label_sample100_path = 'E:/code/datasets/obj_aug_coco1000/select/sample/labels'
    # os.makedirs(img_sample100_path, exist_ok=True)
    # os.makedirs(label_sample100_path, exist_ok=True)
    # random_sample(img_sample_path, label_sample_path, img_sample100_path, label_sample100_path)

    # # 合并三种扩增方式到 Taug
    # insertion_dir = 'E:/code/datasets/coco_test_1000/aug/insertion'
    # remove_dir = 'E:/code/datasets/coco_test_1000/aug/remove'
    # replace_dir = 'E:/code/datasets/coco_test_1000/aug/replace'
    # out_dir = 'E:/code/datasets/coco_test_1000/aug/all'
    # os.makedirs(out_dir, exist_ok=True)
    # get_aug(insertion_dir, remove_dir, replace_dir, out_dir)

    # # 从 Taug 中随机选择 100 个测试用例
    # img_aug_path = 'E:/code/datasets/coco1000/aug/all/images'
    # label_aug_path = 'E:/code/datasets/coco1000/aug/all/labels'
    # img_random_path = 'E:/code/datasets/coco1000/select/random/images'
    # label_random_path = 'E:/code/datasets/coco1000/select/random/labels'
    # os.makedirs(img_random_path, exist_ok=True)
    # os.makedirs(label_random_path, exist_ok=True)
    # random_sample(img_aug_path, label_aug_path, img_random_path, label_random_path, m=1000)

    # 获取按照 mAP从小到达选择的测试用例
    file_path = 'E:/code/yolov5-master/save_paths.npy'
    label_path = 'E:/code/datasets/coco1000/aug/all/labels'
    img_select_path = 'E:/code/datasets/coco1000/select/select_20/images'
    label_select_path = 'E:/code/datasets/coco1000/select/select_20/labels'
    os.makedirs(img_select_path, exist_ok=True)
    os.makedirs(label_select_path, exist_ok=True)
    get_select(file_path, label_path, img_select_path, label_select_path, m=600)
