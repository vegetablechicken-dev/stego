# -*- coding: utf-8 -*-
import math
import random
import cv2  # 用于图像缩放
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.stats import chi2 as chi2_dist
from skimage.metrics import structural_similarity as ssim

def LSB_embed_random(fig, p, bit_level=1):
    '''
    LSB连续嵌入
    :param fig: 待嵌入图片
    :param p: 隐写率
    :param bit_level: 每一像素嵌入多少比特信息
    :return: fig_embedded
    '''
    height, width = fig.shape
    N = height * width
    if math.floor(p) != 0 and p != 1:
        raise Exception('P should be between 0 and 1.')
    if bit_level <= 0 or bit_level >= 8:
        raise Exception('Bit level should be between 0 and 8.')
    num_embed = int(p * N)
    if num_embed == 0:
        return fig.copy()
    # 生成所有像素的索引
    all_indices = list(range(N))
    # 随机选择num_embed个像素
    embed_indices = random.sample(all_indices, num_embed)
    # 生成随机比特序列
    bits = [random.randint(0, 2 ** bit_level - 1) for _ in range(num_embed)]
    
    fig_embedded = fig.copy().flatten()
    mask = (255 >> bit_level) << bit_level
    for idx, bit in zip(embed_indices, bits):
        fig_embedded[idx] = fig_embedded[idx] & mask ^ bit
        
    fig_embedded = fig_embedded.reshape(height, width)
    return fig_embedded

# LSB隐写（连续p*N个像素嵌入）
def LSB_embed_continuous(fig, p, bit_level=1, flag=True):
    '''
    LSB连续嵌入
    :param fig: 待嵌入图片
    :param p: 隐写率
    :param bit_level: 每一像素嵌入多少比特信息
    :param flag: 是否随机选取嵌入的起始坐标，True表示随机选取, False 表示从0开始
    :return: fig_embedded
    '''
    height, width = fig.shape
    N = height * width
    if math.floor(p) != 0 and p != 1:
        raise Exception('P should be between 0 and 1.')
    if bit_level <= 0 or bit_level >= 8:
        raise Exception('Bit level should be between 0 and 8.')
    num_embed = int(p * N)
    if num_embed == 0:
        return fig.copy()
    bits = [random.randint(0, 2 ** bit_level - 1) for _ in range(num_embed)]
    start_idx = random.randint(0, N - num_embed)
    if not flag:
        start_idx = 0

    fig_embedded = fig.copy().flatten()
    mask = (255 >> bit_level) << bit_level
    for idx, bit in zip(list(range(start_idx, start_idx + num_embed)), bits):
        fig_embedded[idx] = fig_embedded[idx] & mask ^ bit
    fig_embedded = fig_embedded.reshape(height, width)
    return fig_embedded

if __name__=='__main__':
    idx = 0
    rates = np.arange(0.1, 1.1, 0.1)
    for i in rates:
        for j in range(1, 11, 1):
            org_img = Image.open('1\\' + str(j) + '.pgm')
            org = np.array(org_img.convert('L'))
            org_img.close()

            fig_embedded = LSB_embed_continuous(org, i, 1, False)
            stg = Image.fromarray(fig_embedded)
            stg.save('2\\' + str(idx) + '.pgm')
            idx += 1

            fig_embedded = LSB_embed_random(org, i, 1)
            stg = Image.fromarray(fig_embedded)
            stg.save('2\\' + str(idx) + '.pgm')
            idx += 1

