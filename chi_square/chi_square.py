# -*- coding: utf-8 -*-
import math
import random
import cv2  # 用于图像缩放
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.stats import chi2 as chi2_dist
from skimage.metrics import structural_similarity as ssim


# 设置固定种子以确保可重复性
random.seed(42)
np.random.seed(42)
plt.rcParams['font.sans-serif']=['SimHei'] # 显示中文

# LSB隐写（随机选择p*N个像素进行嵌入）
def LSB_embed_random(fig, p, bit_level=1):
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
def LSB_embed_continuous(fig, p, bit_level=1):
    # 连续插入LSB
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
    random_idx = random.randint(0, N - num_embed)
    print(bits)
    print(random_idx)

    fig_embedded = fig.copy().flatten()
    mask = (255 >> bit_level) << bit_level
    for idx, bit in zip(list(range(random_idx, random_idx + num_embed)), bits):
        fig_embedded[idx] = fig_embedded[idx] & mask ^ bit
    fig_embedded = fig_embedded.reshape(height, width)
    return fig_embedded

# 图像分块
def image_partition(fig, block_size=(8, 8)):
    height, width = fig.shape
    block_h, block_w = block_size
    blocks = []
    for i in range(0, height, block_h):
        for j in range(0, width, block_w):
            block = fig[i:i + block_h, j:j + block_w]
            # 如果块大小不符合，进行填充
            if block.shape != (block_h, block_w):
                pad_h = block_h - block.shape[0]
                pad_w = block_w - block.shape[1]
                block = np.pad(block, ((0, pad_h), (0, pad_w)), 'constant', constant_values=0)
            blocks.append(block)
    return blocks

# 计算卡方统计量（支持多比特替换检测）
def chi_square_test(original_fig, stego_fig, bit_level=2):
    """
    对比原始图像和隐写图像的像素值分布，计算卡方统计量
    :param original_fig: 原始图像（二维numpy数组）
    :param stego_fig: 隐写图像（二维numpy数组）
    :param bit_level: LSB位数（默认为2）
    :return: 卡方统计量
    """
    # 提取指定位的LSB
    mask = (1 << bit_level) - 1
    original_bits = original_fig & mask
    stego_bits = stego_fig & mask

    # 计算频数
    original_counts = np.bincount(original_bits.flatten(), minlength=2 ** bit_level)
    stego_counts = np.bincount(stego_bits.flatten(), minlength=2 ** bit_level)

    # 期望频数（假设无隐写，即 stego_counts = original_counts）
    expected = original_counts
    observed = stego_counts

    # 避免期望频数为0
    nonzero = expected != 0
    chi2 = np.sum(((observed[nonzero] - expected[nonzero]) ** 2) / expected[nonzero])

    return chi2

def chi_square_pvalue(chi2, bit_level=2):
    """
    计算卡方统计量对应的p值
    :param chi2: 卡方统计量
    :param bit_level: LSB位数
    :return: p值
    """
    df = (2 ** bit_level) - 1  # 自由度
    p_value = 1 - chi2_dist.cdf(chi2, df)
    return p_value

def chi_square(martix):
    '''
    计算卡方统计量对应的p值
    :param matrix: 待进行卡方检验的矩阵
    :return: p值
    '''
    count = np.zeros(256,dtype=int)
    for i in range(len(martix)):
        for j in range(len(martix[0])):
            count[martix[i][j]] += 1
    # H[2i]的观测值
    h2i = count[2:255:2]
    # H[2i]的期望值H[2i] = (H[2i] + H[2i + 1]) / 2
    h2is = (h2i+count[3:256:2])/2
    filter= (h2is!=0)
    k = sum(filter)
    idx = np.zeros(k,dtype=int)
    for i in range(127):
        if filter[i] == True:
            idx[sum(filter[1:i])] = i
    r=sum(((h2i[idx] - h2is[idx])**2) / (h2is[idx]))
    p = 1-chi2_dist.cdf(r,k-1)
    return p


def plot_bit_histogram(original_fig, stego_fig, bit_level=2):
    """
    绘制原始图像和隐写图像的LSB位分布直方图
    :param original_fig: 原始图像
    :param stego_fig: 隐写图像
    :param bit_level: LSB位数
    """
    mask = (1 << bit_level) - 1
    original_bits = original_fig & mask
    stego_bits = stego_fig & mask

    original_counts = np.bincount(original_bits.flatten(), minlength=2 ** bit_level)
    stego_counts = np.bincount(stego_bits.flatten(), minlength=2 ** bit_level)

    labels = [f"{i:0{bit_level}b}" for i in range(2 ** bit_level)]

    x = np.arange(len(labels))  # 标签位置
    width = 0.35  # 柱状图宽度

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, original_counts, width, label='原始图像')
    rects2 = ax.bar(x + width/2, stego_counts, width, label='隐写图像')

    ax.set_xlabel('LSB位值')
    ax.set_ylabel('频数')
    ax.set_title(f'LSB{bit_level}位分布比较')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

# 计算PSNR
def calculate_psnr(original, stego):
    mse = np.mean((original - stego) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# 计算SSIM
def calculate_ssim(original, stego):
    return ssim(original, stego, data_range=stego.max() - stego.min())

# 主流程：仅卡方分析
def main_comparison(bit_level=2):
    # 读取并转换为灰度图像
    img = Image.open('lena_512.bmp').convert('L')
    original_fig = np.array(img)

    # 定义隐写率
    rates = np.arange(0.0, 1.0, 0.1)
    M = 5  # 每种隐写率进行5次
    chi2_results = []

    for p in rates:
        chi2_vals = []
        for m in range(M):
            # 隐写
            fig_embedded = LSB_embed_random(original_fig, p)

            # 卡方测试
            chi2 = chi_square_test(original_fig, fig_embedded, bit_level=bit_level)
            chi2_vals.append(chi2)

        # 计算平均卡方值和标准差
        bar_chi2 = np.mean(chi2_vals)
        sigma_chi2 = np.std(chi2_vals)
        chi2_results.append({'p': p, 'bar_chi2': bar_chi2, 'sigma_chi2': sigma_chi2})

        print(
            f"隐写率p={p:.1f}: 卡方统计量平均={bar_chi2:.4f}, 标准差={sigma_chi2:.4f}")

    # 计算p值
    p_values = [chi_square_pvalue(item['bar_chi2'], bit_level=bit_level) for item in chi2_results]

    # 绘制卡方统计量及p值
    plt.figure(figsize=(12, 6))
    ps_chi2 = [item['p'] for item in chi2_results]
    bar_chi2s = [item['bar_chi2'] for item in chi2_results]
    sigmas_chi2 = [item['sigma_chi2'] for item in chi2_results]
    plt.errorbar(ps_chi2, bar_chi2s, yerr=sigmas_chi2,
                 fmt='o-', ecolor='lightgray', elinewidth=3, capsize=0,
                 label=f'卡方统计量 (bit_level={bit_level})')
    plt.plot(ps_chi2, p_values, 's-', label='p值')
    plt.xlabel('隐写率 p')
    plt.ylabel('卡方统计量 $\chi^2$ / p值')
    plt.title('隐写率 vs 卡方统计量及p值')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制位分布直方图示例（选择 p=0.3 进行展示）
    example_p = 0.3
    fig_embedded_example = LSB_embed_random(original_fig, example_p)
    plot_bit_histogram(original_fig, fig_embedded_example, bit_level=bit_level)

    return chi2_results

# 运行主流程
if __name__ == "__main__":
    # 选择 bit_level=2 进行多比特替换检测
    # chi2_results = main_comparison(bit_level=5)
    a = [list(range(30, 33)) for _ in range(3)]
    a = np.array(a)
    print(LSB_embed_continuous(a, 1, 3))
    
    
    
