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
# random.seed(42)
# np.random.seed(42)
plt.rcParams['font.sans-serif']=['SimHei'] # 显示中文

# LSB隐写（随机选择p*N个像素进行嵌入）
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

# 图像分块
def image_partition(fig, block_size=(8, 8)):
    height, width = fig.shape
    block_h, block_w = block_size
    blocks = []
    for i in range(0, height, block_h):
        for j in range(0, width, block_w):
            block = fig[i:i + block_h, j:j + block_w]
            # 如果块大小不符合，直接返回剩余的块
            if block.shape != (block_h, block_w):
                blocks.append(fig[i:, j:])
                continue
            blocks.append(block)
    return blocks

def chi_square(martix):
    '''
    计算卡方统计量对应的p值
    :param matrix: 待进行卡方检验的矩阵
    :return: r: 统计值; p: 概率值
    小小吐槽：前面同学写的代码似乎有问题，如果我可以获取原始图像，我为什么还需要卡方检测来检测图片是否隐写呢？
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
    return (r, p)

def chi_square_multi_bits(martix, bit_level=1):
    count = np.zeros(256,dtype=int)
    for i in range(len(martix)):
        for j in range(len(martix[0])):
            count[martix[i][j]] += 1
    delta = 2 ** bit_level
    h2i = count[0: 255: delta]
    h2is = np.zeros(256 // delta,dtype=int)
    
    for i in range(delta):
        h2is = (h2is + count[i: 256: delta])
    h2is = h2is / delta
    filter= (h2is!=0)
    k = sum(filter)
    idx = np.zeros(k,dtype=int)
    for i in range(256 // delta):
        if filter[i] == True:
            idx[sum(filter[1:i])] = i
    r=sum(((h2i[idx] - h2is[idx])**2) / (h2is[idx]))
    p = 1-chi2_dist.cdf(r,k-1)
    return (r, p)

def plot_bit_histogram(original_fig, stego_fig, bit_level=1):
    """
    绘制原始图像和隐写图像的像素频数分布直方图
    :param original_fig: 原始图像
    :param stego_fig: 隐写图像
    :param bit_level: LSB位数
    """
    # 防御式编程（其实是懒得改了）
    original_bits = original_fig
    stego_bits = stego_fig

    original_counts = np.bincount(original_bits.flatten(), minlength=256)[50: 90]
    stego_counts = np.bincount(stego_bits.flatten(), minlength=256)[50: 90]

    plt.subplot(211)
    plt.subplots_adjust(hspace=0.3)
    plt.title("灰度图像直方图")
    plt.hist(original_bits.flatten(), bins=np.arange(40, 81, 1),
                rwidth=0.1, align="left")

    plt.xticks(range(40, 81))
    plt.subplot(212)
    plt.title("LSB 隐写图像直方图")
    plt.hist(stego_bits.flatten(), bins=np.arange(40, 81, 1), rwidth=0.1,
                align="left")
    plt.xticks(range(40, 81))
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
    i = Image.open('lena_512.bmp')
    img = i.convert('L')
    original_fig = np.array(img)
    i.close()

    # 定义隐写率
    rates = np.arange(0.0, 1.1, 0.1)
    M = 5  # 每种隐写率进行5次
    chi2_results = []

    for p in rates:
        chi2_vals = []
        p_values = []
        p_val = 0
        for m in range(M):
            # 隐写
            fig_embedded = LSB_embed_continuous(original_fig, p, bit_level, False)

            # 卡方测试
            chi2, p_val = chi_square(fig_embedded)
            chi2_vals.append(chi2)
            p_values.append(p_val)

        # 计算平均卡方值和标准差
        bar_chi2 = np.mean(chi2_vals)
        sigma_chi2 = np.std(chi2_vals)
        chi2_results.append({'p': p, 'bar_chi2': bar_chi2, 'sigma_chi2': sigma_chi2})

        print(
            f"隐写率p={p:.1f}: 卡方统计量平均={bar_chi2:.4f}, 标准差={sigma_chi2:.4f}, 预测存在隐写的概率={p_val:.4f}")

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

# evalution functions
def embed_evalution(flag=True):
    '''
    :param flag: 表明是否是连续嵌入, True为连续, False为随机
    '''
    # 读取并转换为灰度图像
    i = Image.open('lena_512.bmp')
    img = i.convert('L')
    original_fig = np.array(img)
    i.close()

    # 定义隐写率
    rates = np.arange(0.0, 1.1, 0.1)
    M = 5  # 每种隐写率进行5次
    bars_chi2 = []
    sigmas_chi2 = []
    bars_p = []
    for p in rates:
        chi2_vals = []
        p_values = []
        p_val = 0
        for m in range(M):
            if flag:
                fig_embedded = LSB_embed_continuous(original_fig, p, 1, False)
            else:
                fig_embedded = LSB_embed_random(original_fig, p, 1)
            partitions = image_partition(fig_embedded, (fig_embedded.shape[0] // 4, fig_embedded.shape[1]))
            tmp_chi2 = []
            tmp_p = []
            for part in partitions:
                chi2, p_val = chi_square(part)
            # print(chi2, p_val)
                tmp_chi2.append(chi2)
                tmp_p.append(p_val)
            chi2_vals.append(np.min(tmp_chi2))
            p_values.append(np.max(tmp_p))
        bar_chi2 = np.mean(chi2_vals)
        bars_chi2.append(bar_chi2)
        bar_p = np.mean(p_values)
        bars_p.append(bar_p)
        print(f"隐写率p={p:.1f}: 卡方统计量={bar_chi2:.4f}, 预测存在隐写的概率={bar_p:.4f}")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(rates, bars_chi2)
    plt.title("隐写率 vs 卡方统计量")
    plt.xlabel("隐写率 p")
    plt.ylabel("$\chi^2$统计量")
    plt.subplot(212)
    plt.plot(rates, bars_p)
    plt.title("隐写率 vs 预测概率")
    plt.xlabel("隐写率 p")
    plt.ylabel("预测概率")
    plt.grid()
    plt.show()

def evalution_sample_size():
    # i = Image.open('lena_512.bmp')
    # img = i.convert('L')
    img = Image.open('1\\7.pgm')
    original_fig = np.array(img)
    img.close()
    # 设定隐写率为0.5
    p = 0.5
    fig_embedded = LSB_embed_continuous(original_fig, p, 1)
    x = np.arange(1, 11, 1)
    chi2s_row = [[], []]
    chi2s_col = [[], []]
    M = 5
    # 修改图片分块列的大小
    for i in x:
        chi2_vals = []
        p_values = []
        for m in range(M):
            fig_embedded = LSB_embed_continuous(original_fig, p, 1, True)
            k = i / 10.0
            partitions = image_partition(fig_embedded, (fig_embedded.shape[0], int(fig_embedded.shape[1] * k)))
            tmp_chi2 = []
            tmp_p = []
            for part in partitions:
                chi2, p_val = chi_square(part)
                # print(chi2, p_val)
                tmp_chi2.append(chi2)
                tmp_p.append(p_val)
            chi2_vals.append(np.min(tmp_chi2))
            p_values.append(np.max(tmp_p))
        bar_chi2 = np.mean(chi2_vals)
        bar_p = np.mean(p_values)
        chi2s_col[0].append(bar_chi2)
        chi2s_col[1].append(bar_p)
        print(f"图片分块列大小占原始图片的比例k={k:.1f}: 卡方统计量={bar_chi2:.4f}, 预测存在隐写的概率={bar_p:.4f}")
    print("=" * 64)
    for i in x:
        chi2_vals = []
        p_values = []
        for m in range(M):
            fig_embedded = LSB_embed_continuous(original_fig, p, 1, True)
            k = i / 10.0
            partitions = image_partition(fig_embedded, (int(fig_embedded.shape[0] * k), fig_embedded.shape[1]))
            tmp_chi2 = []
            tmp_p = []
            for part in partitions:
                chi2, p_val = chi_square(part)
                # print(chi2, p_val)
                tmp_chi2.append(chi2)
                tmp_p.append(p_val)
            chi2_vals.append(np.min(tmp_chi2))
            p_values.append(np.max(tmp_p))
        bar_chi2 = np.mean(chi2_vals)
        bar_p = np.mean(p_values)
        chi2s_row[0].append(bar_chi2)
        chi2s_row[1].append(bar_p)
        print(f"图片分块行大小占原始图片的比例k={k:.1f}: 卡方统计量={bar_chi2:.4f}, 预测存在隐写的概率={bar_p:.4f}")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(x / 10, chi2s_row[1], marker='o')
    plt.title("图片分块行大小占原始图片的比例 vs 预测存在隐写的概率")
    plt.xlabel("图片分块行大小占原始图片的比例")
    plt.ylabel("预测存在隐写的概率")
    plt.grid()
    plt.subplot(212)
    plt.plot(x / 10, chi2s_col[1], marker='o')
    plt.title("图片分块列大小占原始图片的比例 vs 预测存在隐写的概率")
    plt.xlabel("图片分块列大小占原始图片的比例")
    plt.ylabel("预测存在隐写的概率")
    plt.grid()
    plt.show()



# 运行主流程
if __name__ == "__main__":
    # fig1: 原始和隐写图像像素值频率直方图对比
    # org_img1 = Image.open('lena_512.bmp').convert('L')
    # org_img = np.array(org_img1)
    # org_img1.close()
    # stego_img = LSB_embed_random(org_img, 1)
    # plot_bit_histogram(org_img, stego_img)

    # evalution1: 连续嵌入，考察预测的概率值随隐写率变化，以及准确率、误报率、漏报率影响
    # embed_evalution()
    print('=' * 64)
    # evalution2: 随机嵌入，考察预测的概率值随隐写率变化，以及准确率、误报率、漏报率影响
    # embed_evalution(False)
    print('=' * 64)
    # evalution3: 考虑多比特嵌入
    org_img1 = Image.open('lena_512.bmp').convert('L')
    org_img = np.array(org_img1)
    org_img1.close()
    stego_img = LSB_embed_continuous(org_img, 0.6, 2)
    p = image_partition(stego_img, (int(stego_img.shape[0] * 0.6), stego_img.shape[1]))
    print(chi_square_multi_bits(stego_img, 1))
    print(chi_square_multi_bits(p[0], 2))
    
    print('=' * 64)
    # evalution4: 考虑sample of size，以及准确率、误报率、漏报率影响
    # evalution_sample_size()
    # evalution5: 改变卡方分析的阈值，考察准确率、误报率、漏报率
    # evalution6: 连续嵌入，与RS算法和XX算法比较
    # evalution7: 随机嵌入，与RS算法和XX算法比较
    
    

