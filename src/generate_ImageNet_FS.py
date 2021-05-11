import os
import tqdm
import json
import csv
import shutil

train_folder = f"{os.path.expanduser('~')}/Datasets/ILSVRC2012_train"

# for img in tqdm.tqdm(os.listdir(train_folder)):
#     if '[' in img:
#         shutil.rmtree(f"{train_folder}/{img}")


    # import ipdb; ipdb.set_trace()
    # img_file = img[1:-1].split(',')[0].strip()[1:-1]
    # img_name = img[1:-1].split(',')[1].strip()[1:-1]
    # if not os.path.isdir(f"{train_folder}/{img_file}"):
    #     os.mkdir(f"{train_folder}/{img_file}")
    # os.replace(f"{train_folder}/{img}/{img_file}_{img_name}", f"{train_folder}/{img_file}/{img_file}_{img_name}")

# print(f"There are {i} images in the folder")


# --------------------------------------------------------------- #
# Make base_1, novel_1, base_2, novel_2 csv file for imageNet-FS
# --------------------------------------------------------------- #
header = ['filename','label']

with open('./src/ImageNet_FS.txt', 'r') as f:
    im_fs = json.load(f)

label_names = im_fs['label_names']
novel_classes_1 = im_fs['novel_classes_1']
novel_classes_2 = im_fs['novel_classes_2']
base_classes_1 = im_fs['base_classes_1']
base_classes_2 = im_fs['base_classes_2']

# ----------------------------------------------------------- #
# novel_1.csv
# ----------------------------------------------------------- #
novel_1 = open('./split/fs/novel_1.csv', mode='w')
novel_1_writer = csv.writer(novel_1, delimiter=',')
novel_1_writer.writerow(header)

for novel in novel_classes_1:
    novel_name = label_names[novel]
    for img in os.listdir(f"{train_folder}/{novel_name}"):
        novel_1_writer.writerow([img,novel_name])
novel_1.close()

# # ----------------------------------------------------------- #
# # novel_2.csv
# # ----------------------------------------------------------- #
# novel_2 = open('./split/fs/novel_2.csv', mode='w')
# novel_2_writer = csv.writer(novel_2, delimiter=',')
# novel_2_writer.writerow(header)

# for novel in novel_classes_2:
#     novel_name = label_names[novel]
#     for img in os.listdir(f"{train_folder}/{novel_name}"):
#         novel_2_writer.writerow([img,novel_name])
# novel_2.close()

# # ----------------------------------------------------------- #
# # base_1.csv
# # ----------------------------------------------------------- #
# base_1 = open('./split/fs/base_1.csv', mode='w')
# base_1_writer = csv.writer(base_1, delimiter=',')
# base_1_writer.writerow(header)

# for novel in base_classes_1:
#     novel_name = label_names[novel]
#     for img in os.listdir(f"{train_folder}/{novel_name}"):
#         base_1_writer.writerow([img,novel_name])
# base_1.close()

# # ----------------------------------------------------------- #
# # base_2.csv
# # ----------------------------------------------------------- #
# base_2 = open('./split/fs/base_2.csv', mode='w')
# base_2_writer = csv.writer(base_2, delimiter=',')
# base_2_writer.writerow(header)

# for novel in base_classes_2:
#     novel_name = label_names[novel]
#     for img in os.listdir(f"{train_folder}/{novel_name}"):
#         base_2_writer.writerow([img,novel_name])
# base_2.close()





