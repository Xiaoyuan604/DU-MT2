import os
import nrrd
import h5py
import numpy as np
from tqdm import tqdm

data_dir = "/home/qwe/data/wenxiaoyuan/Data/fengwodata"
h5_save_dir = os.path.join(data_dir, "h5data")

# 如果h5_save_dir不存在，则创建该目录
if not os.path.exists(h5_save_dir):
    os.makedirs(h5_save_dir)

def find_max_boundaries(folders):
    all_minx, all_maxx = [], []
    all_miny, all_maxy = [], []
    all_minz, all_maxz = [], []

    for folder in tqdm(folders):
        label_path = os.path.join(folder, 'Segmentation.seg.nrrd')
        label, _ = nrrd.read(label_path)

        # 获取标签的非零区域
        tempL = np.nonzero(label)
        all_minx.append(np.min(tempL[0]))
        all_maxx.append(np.max(tempL[0]))
        all_miny.append(np.min(tempL[1]))
        all_maxy.append(np.max(tempL[1]))
        all_minz.append(np.min(tempL[2]))
        all_maxz.append(np.max(tempL[2]))

    # 计算所有标签的最大边界
    minx = min(all_minx)
    maxx = max(all_maxx)
    miny = min(all_miny)
    maxy = max(all_maxy)
    minz = min(all_minz)
    maxz = max(all_maxz)

    return minx, maxx, miny, maxy, minz, maxz

def convert_to_h5(folders, minx, maxx, miny, maxy, minz, maxz, h5_base_dir='/home/qwe/data/wenxiaoyuan/Data/h5_data'):
    os.makedirs(h5_base_dir, exist_ok=True)  # 创建保存 h5 文件的主目录
    for idx, folder in enumerate(tqdm(folders), start=1):
        image_path = os.path.join(folder, 'output_image.nrrd')
        label_path = os.path.join(folder, 'Segmentation.seg.nrrd')

        # 读取图像和标签
        image, _ = nrrd.read(image_path)
        label, _ = nrrd.read(label_path)

        # 对图像进行标准化
        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)

        # 裁剪图像和标签到非零区域的边界
        cropped_image = image[minx:maxx, miny:maxy, minz:maxz]
        cropped_label = label[minx:maxx, miny:maxy, minz:maxz]

        # 输出裁剪后数据的尺寸
        print(f"Folder: {folder}")
        print(f"Cropped image shape: {cropped_image.shape}")
        print(f"Cropped label shape: {cropped_label.shape}\n")

        # 创建对应的文件夹 1-100
        folder_name = str(idx)
        h5_folder_dir = os.path.join(h5_base_dir, folder_name)
        os.makedirs(h5_folder_dir, exist_ok=True)

        # 保存为 h5 文件，并命名为 image.h5
        h5_path = os.path.join(h5_folder_dir, 'image.h5')
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('image', data=cropped_image, compression="gzip")
            f.create_dataset('label', data=cropped_label, compression="gzip")

if __name__ == '__main__':
    # 获取所有子文件夹路径
    folders = [os.path.join(data_dir, str(i)) for i in range(1, 101)]

    # 计算所有标签的最大边界
    minx, maxx, miny, maxy, minz, maxz = find_max_boundaries(folders)

    # 转换数据并保存为 h5 格式
    convert_to_h5(folders, minx, maxx, miny, maxy, minz, maxz)



