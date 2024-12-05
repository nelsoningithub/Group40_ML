import os
import random
import string
from PIL import Image, ImageDraw, ImageFont
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torchvision  # Add this line
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import shutil
from torchvision.transforms.functional import to_pil_image
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from skimage.morphology import dilation, square
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np

def calculate_accuracy_with_only_watermark_region(clean_images_dir, watermarked_images_dir, processed_images_dir, mask_dir, similarity_threshold=0.9):
    """
    計算去浮水印模型的正確率，只考慮浮水印及其周圍區域，不考慮其他區域。

    :param clean_images_dir: 原始圖片文件夾路徑
    :param processed_images_dir: 去浮水印圖片文件夾路徑
    :param mask_dir: 浮水印區域掩膜文件夾路徑，該掩膜標識浮水印位置（1表示浮水印區域，0表示非浮水印區域）
    :param similarity_threshold: SSIM 判斷閾值
    :return: 正確率，平均 SSIM 分數
    """
    clean_images = natsorted(os.listdir(clean_images_dir))
    watermarked_images = natsorted(os.listdir(watermarked_images_dir))
    processed_images = natsorted(os.listdir(processed_images_dir))
    masks = natsorted(os.listdir(mask_dir))

    if len(clean_images) != len(processed_images) or len(clean_images) != len(masks):
        raise ValueError("圖片數量或掩膜數量不一致！")

    total_images = len(clean_images)
    correct_count = 0
    total_score = 0
    time = 6

    for clean_image, watermarked_image, processed_image, mask_name in zip(clean_images, watermarked_images, processed_images, masks):
        # 讀取圖片和掩膜
        clean_img = imread(os.path.join(clean_images_dir, clean_image))
        watermarked_img = imread(os.path.join(watermarked_images_dir, watermarked_image))
        processed_img = imread(os.path.join(processed_images_dir, processed_image))
        mask = imread(os.path.join(mask_dir, mask_name))

        # 掩膜區域：1 表示浮水印區域
        watermark_region = mask > 0.4

        # 計算水印區域的矩形框
        coords = np.argwhere(watermark_region)
        if coords.size == 0:
            print(f"掩膜 {mask_name} 中沒有標註浮水印區域，跳過此影像！")
            total_images -= 1  # 減少總影像計數
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        rectangular_mask = np.zeros_like(mask, dtype=bool)
        rectangular_mask[y_min:y_max+1, x_min:x_max+1] = True

        clean_img_masked = clean_img[y_min:y_max+1, x_min:x_max+1]
        processed_img_masked = processed_img[y_min:y_max+1, x_min:x_max+1]
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # 顯示調試用的矩形框掩
        if time > 0:
            # 顯示水印圖片
            ax[0].imshow(clean_img_masked)
            ax[0].set_title("Clean_img_masked")
            ax[0].axis('off')
            # 顯示矩形掩膜
            ax[1].imshow(watermarked_img, cmap='gray')
            ax[1].set_title("Watermarked_img")
            ax[1].axis('off')
            # 顯示處理後的圖片
            ax[2].imshow(processed_img_masked)
            ax[2].set_title("Processed_img_masked")
            ax[2].axis('off')
            plt.pause(0.5)  # 暫停 0.5 秒，模擬實時更新
            time -= 1

        data_range = clean_img.max() - clean_img.min()
        # 計算 SSIM
        score, _ = ssim(
            clean_img_masked, 
            processed_img_masked, 
            channel_axis=-1,  # 指定最後一軸是顏色通道
            data_range=data_range, 
            full=True,
        )
        if score >= similarity_threshold:
            correct_count += 1
        total_score += score

        # 清空圖形窗口
        plt.close('all')

    accuracy = correct_count / total_images
    average_score = total_score / total_images  # 計算平均 SSIM 分數
    return accuracy, average_score

# Step 3: 設定圖片文件夾路徑並計算正確率
# clean_images_dir = "web_dataset_split/test2/no_watermark"  # 原始圖片路徑
# watermarked_images_dir = "web_dataset_split/test2/watermarked"  # 原始浮水印圖片路徑
# processed_images_dir = "web_test_model_result/watermark_removal_output"  # 浮水印結果路徑
# mask_dir = "web_dataset_split/test2/masks"  # 浮水印區域掩膜的路徑

clean_images_dir = "dataset_split/test2/no_watermark"  # 原始圖片路徑
watermarked_images_dir = "dataset_split/test2/watermarked"  # 原始浮水印圖片路徑
processed_images_dir = "test_model_result/watermark_removal_output"  # 浮水印結果路徑
mask_dir = "dataset_split/test2/masks"  # 浮水印區域掩膜的路徑

similarity_threshold = 0.7  # 設置 SSIM 判斷閾值

accuracy, average_score = calculate_accuracy_with_only_watermark_region(
    clean_images_dir, watermarked_images_dir, processed_images_dir, mask_dir, similarity_threshold
)

print(f"去浮水印正確率: {accuracy * 100:.2f}%")
print(f"平均 SSIM 分數: {average_score:.4f}")
