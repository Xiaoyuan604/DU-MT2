import nrrd
import numpy as np
import SimpleITK as sitk

# 读取 nrrd 文件
image = sitk.ReadImage('/home/qwe/data/wenxiaoyuan/Data/fengwodata/1/output_image.nrrd')
# 查看分辨率
# 获取各方向的分辨率
spacing = image.GetSpacing()
print("Spacing directions (in mm):")
for i, s in enumerate(spacing):
    print(f"Direction {i}: {s} mm")