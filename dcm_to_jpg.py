import os
import pydicom
import cv2
import pandas as pd
import multiprocessing
from PIL import Image

def convert_dcm_to_jpg(file_path, imsize=None, counter=None, lock=None):
    """ 将 DICOM 文件转换为 JPEG 并保存，同时增加计数 """
    try:
        ds = pydicom.dcmread(file_path)
        img_array = ds.pixel_array

        # 应用 RescaleIntercept 和 RescaleSlope
        if 'RescaleIntercept' in ds and 'RescaleSlope' in ds:
            img_array = img_array * ds.RescaleSlope + ds.RescaleIntercept

        # 调整大小
        if imsize is not None:
            img_array = cv2.resize(img_array, (imsize, imsize))

        # 归一化到 [0, 255]
        if img_array.max() > 0:
            img_array = cv2.convertScaleAbs(img_array, alpha=(255.0 / img_array.max()))

        # 如果是 MONOCHROME1，需要取反
        if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
            img_array = cv2.bitwise_not(img_array)

        # 转换为 RGB 并保存
        img = Image.fromarray(img_array).convert("RGB")
        jpg_path = os.path.splitext(file_path)[0] + ".jpg"
        img.save(jpg_path)

        # 计数并显示进度
        with lock:  # 使用锁保证安全更新
            counter.value += 1
            if counter.value % 10 == 0:
                print(f"已处理 {counter.value} 张 DICOM 文件")

    except Exception as e:
        print(f"转换失败: {file_path}, 错误: {e}")

def process_row(row, counter, lock):
    """ 处理 CSV 中的一行数据，转换所有 DICOM 文件 """
    path_list = ['c_view_cc', 'c_view_mlo', '2d_cc', '2d_mlo']
    for elem in path_list:
        dcm_path = row[elem]
        if pd.notna(dcm_path):  # 确保路径有效
            convert_dcm_to_jpg(dcm_path, imsize=224, counter=counter, lock=lock)

if __name__ == "__main__":
    df = pd.read_csv("combine.csv")

    # 共享变量：计数器 & 线程锁
    manager = multiprocessing.Manager()
    counter = manager.Value("i", 0)  # 初始化计数器
    lock = manager.Lock()  # 线程锁，防止多个进程同时修改 counter

    # 启动多进程池
    num_workers = min(8, os.cpu_count())  # 最多使用 8 个 CPU 核心
    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(process_row, [(df.iloc[i], counter, lock) for i in range(len(df))])

    print(f"所有 DICOM 文件转换完成！总共处理 {counter.value} 张图片。")
