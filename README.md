# ObjTest: Object-Level Mutation for Testing Object Detection Systems 

ObjTest has conducted experiments on 3 datasets (COCO, VOC, BDD100K) and the testing subject is YOLOv5 with 3 different model sizes (YOLOv5s, YOLOv5m, YOLOv5l).

## Environment

- python=3.9
- Rembg

We recommend using conda to configure the environment:

```sh
conda create -n python=3.9
conda activate objtest
pip install rembg
```

## Object-level Transformation

We have implemented three object-level transformations: object insertion, removing, replacing.

### Object Insertion

- **Step1**: Extract objects and remove their background:

```sh
python remove_bg.py --dataset coco
```

Manually selecte some clear and clean images and save to  ``ObjTest/coco_classes`` folder. In our demo repository, the object images have already saved in this folder. 

- **Step2**: Run ``img_add.py`` to overlay the mask image onto the original image and save the new image and the corresponding label:

````sh
python img_add.py --task insertion --dataset coco --type val
````

The generated datasets after executing the insertion operator are in the ``E:/code/datasets/obj_aug_coco1000/insertion/images`` and ``E:/code/datasets/obj_aug_coco1000/insertion/ labels`` folders, they can be input to YOLOv5 directly. You can change the file path in the code yourself.

### Object Removing

- **Step1**: Run ``ObjTest/gen_mask_img.py`` to generate the mask image corresponding to COCO1000 and the label after removing:

````sh
python gen_mask_img.py --task remove --dataset coco --type val
````

- **Step2**: Download and configure the [lama](Download and configure the lama project) project. Then, go to the ``lama-main`` project and perform the fix for mask. You can change the floaer path by yourself.

````sh
cd lama-main
conda activate lama
# coco val:
python ./bin/predict.py model.path=E:/code/lama-main/big-lama indir=E:/code/datasets/obj_aug_coco1000/remove/images_tmp outdir=E:/code/datasets/obj_aug_coco1000/remove/images
````

The generated datasets after executing the remove operator are in the ``E:/code/datasets/obj_aug_coco1000/remove/images`` and ``E:/code/datasets/obj_aug_coco1000/remove/labels`` folders, they can be input to YOLOv5 directly.

### Object Replacing

- **Step1**: prepare the mask image:

````sh
python gen_mask_img.py --task replace --dataset coco --type val
````

After running, ``datasets/obj_aug_coco1000/replace/images_tmp1`` contains the original map and the corresponding mask location map; ``datasets/obj_aug_coco1000/replace/labels_tmp`` contains the new label after removing object from the original image and the deleted object's label file.

- **Step2**: Use ``lama`` to repair the removed object image. You need to change the path location yourself.

````sh
cd lama-main
conda activate lama
# coco val:
python ./bin/predict.py model.path=E:/code/lama-main/big-lama indir=E:/code/datasets/obj_aug_coco1000/replace/images_tmp1 outdir=E:/code/datasets/obj_aug_coco1000/replace/images_tmp2
````

最后，用 img_add.py增加 Obj，需要在``datasets/obj_aug_coco1000/replace/labels_tmp2``的基础上增加Obj，结合``datasets/obj_aug_coco1000/replace/labels_tmp/xxx_replace.txt``提供的类别和位置信息：

- **Step3**: Add object with ``img_add.py``. It requires adding Object to ``datasets/obj_aug_coco1000/replace/labels_tmp2``, combined with ``datasets/obj_aug_coco1000/replace/labels_tmp /xxx_replace.txt`` to provide the category and location information.

````sh
python img_add.py --task replace --dataset coco --type val
````

The generated datasets after executing the replace operator are in the ``E:/code/datasets/obj_aug_coco1000/replace/images`` and ``E:/code/datasets/obj_aug_coco1000/replace/labels`` folders, they can be input to YOLOv5 directly.

## Verification in YOLOv5 OD System

In the YOLOv5 project, verify each of the generated dataset with three transformations:

````sh
conda activate yolov5
python val.py --weights ./torch_models/yolov5s.pt --data coco_insertion.yaml --img 640 --batch-size 4 --verbose
python val.py --weights ./torch_models/yolov5s.pt --data coco_remove.yaml --img 640 --batch-size 4 --verbose
python val.py --weights ./torch_models/yolov5s.pt --data coco_replace.yaml --img 640 --batch-size 4 --verbose
````

## Baselines

- **Baseline1**: The first type of baseline, which directly performs random transformations of brightness, contrast, and Gaussian blur on the image as a whole, without changing the label information.

````sh
python baseline1_aug.py --img_dir ../datasets/coco1000/images/val2017 --label_dir ../datasets/coco1000/labels/val2017 --save_dir ../datasets/obj_aug_coco1000
````

- **Baseline2**: The second type of baseline, cutting and stitching of images, requires changing the label information.

````sh
python baseline2_aug.py --img_dir ../datasets/coco1000/images/val2017 --label_dir ../datasets/coco1000/labels/val2017 --save_dir ../datasets/obj_aug_coco1000
````

## Naturalness verification of generated images

In the ``pytorch-fid`` folder, the FID score is calculated by inputing in the path of two image folders; the smaller the FID score, the more realistic and natural the image is.

````sh
cd pytorch_fid
conda activate fid
run_coco.sh
````

