import h5py

# 假设h5文件路径为 'data.h5'
with h5py.File('/home/qwe/data/wenxiaoyuan/DU-UA-MT/UA-MT-master/data/2024_Honeycomb_Seg/1/image.h5', 'r') as f:
    # 查看数据集中的所有键
    print(list(f.keys()))

    # 假设图像数据存储在 'image' 键下
    image_data = f['image'][:]

    # 查看图像数据的形状
    print("Image shape:", image_data.shape)
