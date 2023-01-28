import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default='E:/code/datasets/coco1000/images/val2017', required=True)
    parser.add_argument('--root_path', default='E:/code/datasets/coco1000', required=True)
    parser.add_argument('--txt_name', default='coco1000_val2017.txt', required=True)
    args = parser.parse_args()

    save_li = []
    dir_path = args.img_path.split(args.root_path)[1]
    path_li = os.listdir(args.img_path)
    for p in path_li:
        # save_li.append('./' + dir_path + '/' + p)
        save_li.append('.' + dir_path + '/' + p)

    save_path = os.path.join(args.root_path, args.txt_name)
    with open(save_path, 'w') as f:
        f.writelines([line + '\n' for line in save_li])