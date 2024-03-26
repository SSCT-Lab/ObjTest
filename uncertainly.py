import numpy as np
import os
import shutil

def label_correlation_sorted(file_path, adj):
    pic_info={}
    for filename in os.listdir(file_path):
        with open(file_path+"\\"+filename, 'r') as f:
            label_path=file_path+"\\"+filename
            print("label_path:")
            print(label_path)
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
            max_conf_label=lb[-1][0]
            adj_lb=adj[int(max_conf_label)]
            min_cor=1
            for x in lb:
                min_cor=min(adj_lb[int(x[0])],min_cor)
            pic_info[filename]=min_cor
    return(pic_info)


def coffidence_sorted(file_path):
    pic_info = {}
    for filename in os.listdir(file_path):
        with open(file_path+"\\"+filename, 'r') as f:
            sum=0
            label_path=file_path+"\\"+filename
            print("label_path:")
            print(label_path)
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
            for x in lb:
                sum+=x[-1]
            avg_conf=sum/len(lb)
            pic_info[filename]=avg_conf
    return pic_info




if __name__ == '__main__':
    # adj = np.load("./adj.npy")
    # label_path = r"C:\Users\Xu Jiali\Desktop\南软\测试拓展\ODdata\labels"
    # data_path = r"D:\dataset\coco\aug\labels"
    # dict = label_correlation_sorted(label_path , adj)
    # dict_1 = sorted(dict.items(),key=lambda x: x[1])
    # print(dict_1)
    # with open("./sorted/coco/5l/label_corr.txt","w") as f:
    #     for x in os.listdir(data_path):
    #         if x not in os.listdir(label_path):
    #             f.write("./aug/images/" + x.split('.')[0] + ".png" + '\n')
    #     for x in dict_1:
    #         f.write("./aug/images/" + str((x[0].split('.'))[0]) + ".png" +'\n')
    # f.close()


    # dict=coffidence_sorted(label_path)
    # dict_2 = sorted(dict.items(),key=lambda x: x[1])
    # print(dict_2)
    # with open("./sorted/coco/5m_avg_conf.txt", "w") as f:
    #     for x in os.listdir(data_path):
    #         if x not in os.listdir(label_path):
    #             f.write("./aug/images/" + x.split('.')[0] + ".png" + '\n')
    #     for x in dict_2:
    #         f.write("./aug/images/" + str((x[0].split('.'))[0]) + ".png" +'\n')
    # f.close()



    with open(r"./sorted/bdd/5m_avg_conf.txt", "r") as f:
        lines=f.readlines()
        path=r"D:\dataset\bdd\aug"
        i=0
        while i <= 600:  # 150, 450, 600
            filename=lines[i].split('/')[3]
            filename=filename.split('.')[0]

            image_path=path+"\images\\"+filename+".png"
            image_new_path = r".\600\bdd\images"
            os.makedirs(image_new_path, exist_ok=True)
            label_path=path+"\labels\\"+filename+".txt"
            label_new_path=r".\600\bdd\labels"
            os.makedirs(label_new_path, exist_ok=True)
            shutil.copy(image_path, image_new_path)
            shutil.copy(label_path, label_new_path)
            i+=1










