from rembg import remove
import os
import argparse


def remove_bg(input_dir, output_dir):
    for img_file in os.listdir(input_dir):
        fname = img_file.split('.jpg')[0]
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, fname + '.png')  # 保存透明背景的图片为 png

        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input = i.read()
                output = remove(input)
                o.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['coco', 'openimages', 'voc'], required=True)
    arg = parser.parse_args()

    input_dir, output_dir = '', ''
    if arg.dataset == 'coco':
        input_dir = './coco_last500'  # 原始图片的位置
        output_dir = './coco_last500_remove'  # 去除背景后，透明背景底的图片存储位置
    elif arg.dataset == 'openimages':
        input_dir = './openimages_random500'
        output_dir = './openimages_random500_remove'
    elif arg.dataset == 'voc':
        input_dir = './voc_random500'
        output_dir = './voc_random500_remove'
    else:
        print("This dataset is not currently supported!")

    os.makedirs(output_dir, exist_ok=True)
    remove_bg(input_dir, output_dir)
