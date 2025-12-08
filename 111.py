import os
import sys
import torch
import numpy as np
import time
from tqdm import tqdm

# 设置路径
os.chdir(r'D:\桌面\DEA-Net')
sys.path.append(r'D:\桌面\DEA-Net\code')

print("=== 完整测试SOTS-outdoor数据集 ===")

# 1. 加载模型
from model.backbone import Backbone
from utils.metric import val_psnr, val_ssim
from utils.utils import pad_img

model_path = r'D:\桌面\DEA-Net\trained_models\OTS\PSNR3659_SSIM9897.pth'
print(f"加载模型: {model_path}")

model = Backbone(base_dim=32)
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()
model.cuda()

# 2. 自定义数据集类
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image


class SOTSOutdoorDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.hazy_files = sorted(os.listdir(hazy_dir))
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.hazy_files)

    def __getitem__(self, idx):
        hazy_filename = self.hazy_files[idx]
        clear_filename = hazy_filename.split('_')[0] + '.png'

        hazy_path = os.path.join(self.hazy_dir, hazy_filename)
        clear_path = os.path.join(self.clear_dir, clear_filename)

        # 加载图像
        hazy_img = Image.open(hazy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')

        # 确保尺寸一致
        if hazy_img.size != clear_img.size:
            clear_img = clear_img.resize(hazy_img.size, Image.BILINEAR)

        # 转换为张量
        hazy = self.to_tensor(hazy_img)
        clear = self.to_tensor(clear_img)

        return {
            'hazy': hazy,
            'clear': clear,
            'filename': hazy_filename
        }


# 3. 准备数据集
dataset_path = r'D:\桌面\DEA-Net\dataset\sOTS\outdoor'
hazy_path = os.path.join(dataset_path, 'hazy')
clear_path = os.path.join(dataset_path, 'clear')

dataset = SOTSOutdoorDataset(hazy_path, clear_path)
print(f"数据集大小: {len(dataset)} 张图像")

# 4. 完整测试
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

psnr_values = []
ssim_values = []
filenames = []
inference_times = []

print("\n开始完整测试...")
start_time = time.time()

for batch in tqdm(dataloader, desc="测试进度"):
    hazy = batch['hazy'].cuda()
    clear = batch['clear'].cuda()
    filename = batch['filename'][0]

    with torch.no_grad():
        # 记录推理时间
        torch.cuda.synchronize()
        infer_start = time.time()

        # 应用padding
        H, W = hazy.shape[2], hazy.shape[3]
        hazy_padded = pad_img(hazy, 4)
        output = model(hazy_padded)
        output = torch.clamp(output, 0, 1)
        output = output[:, :, :H, :W]

        torch.cuda.synchronize()
        infer_time = time.time() - infer_start

        # 计算指标
        psnr = val_psnr(output, clear)
        ssim = val_ssim(output, clear).item()

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        filenames.append(filename)
        inference_times.append(infer_time)

total_time = time.time() - start_time

# 5. 计算结果
print(f"\n=== 完整测试结果 ===")
print(f"测试图像总数: {len(psnr_values)}")
print(f"总耗时: {total_time:.2f} 秒")
print(f"平均每张图像耗时: {np.mean(inference_times) * 1000:.2f} ms")
print(f"平均推理时间: {np.mean(inference_times) * 1000:.2f} ms")

print(f"\nPSNR统计:")
print(f"  平均PSNR: {np.mean(psnr_values):.4f} dB")
print(f"  中位数PSNR: {np.median(psnr_values):.4f} dB")
print(f"  标准差: {np.std(psnr_values):.4f} dB")
print(f"  最小值: {np.min(psnr_values):.4f} dB")
print(f"  最大值: {np.max(psnr_values):.4f} dB")

print(f"\nSSIM统计:")
print(f"  平均SSIM: {np.mean(ssim_values):.6f}")
print(f"  中位数SSIM: {np.median(ssim_values):.6f}")
print(f"  标准差: {np.std(ssim_values):.6f}")
print(f"  最小值: {np.min(ssim_values):.6f}")
print(f"  最大值: {np.max(ssim_values):.6f}")

print(f"\n论文报告值:")
print(f"  PSNR: 36.59 dB")
print(f"  SSIM: 0.9897")

# 6. 保存详细结果
os.makedirs('sots_full_results', exist_ok=True)

# 保存每张图像的结果
with open('sots_full_results/detailed_results.csv', 'w') as f:
    f.write("filename,psnr,ssim,inference_time_ms\n")
    for i in range(len(filenames)):
        f.write(f"{filenames[i]},{psnr_values[i]:.4f},{ssim_values[i]:.6f},{inference_times[i] * 1000:.2f}\n")

# 保存统计结果
with open('sots_full_results/summary.txt', 'w') as f:
    f.write("=== SOTS-outdoor 完整测试结果 ===\n")
    f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"模型: {model_path}\n")
    f.write(f"数据集: {dataset_path}\n")
    f.write(f"图像数量: {len(psnr_values)}\n\n")

    f.write("PSNR统计:\n")
    f.write(f"  平均PSNR: {np.mean(psnr_values):.4f} dB\n")
    f.write(f"  中位数PSNR: {np.median(psnr_values):.4f} dB\n")
    f.write(f"  标准差: {np.std(psnr_values):.4f} dB\n")
    f.write(f"  最小值: {np.min(psnr_values):.4f} dB\n")
    f.write(f"  最大值: {np.max(psnr_values):.4f} dB\n\n")

    f.write("SSIM统计:\n")
    f.write(f"  平均SSIM: {np.mean(ssim_values):.6f}\n")
    f.write(f"  中位数SSIM: {np.median(ssim_values):.6f}\n")
    f.write(f"  标准差: {np.std(ssim_values):.6f}\n")
    f.write(f"  最小值: {np.min(ssim_values):.6f}\n")
    f.write(f"  最大值: {np.max(ssim_values):.6f}\n\n")

    f.write("性能统计:\n")
    f.write(f"  总耗时: {total_time:.2f} 秒\n")
    f.write(f"  平均每张图像耗时: {np.mean(inference_times) * 1000:.2f} ms\n\n")

    f.write("论文对比:\n")
    f.write(f"  论文PSNR: 36.59 dB\n")
    f.write(f"  实测PSNR: {np.mean(psnr_values):.4f} dB\n")
    f.write(f"  差异: {36.59 - np.mean(psnr_values):.4f} dB\n\n")

    f.write(f"  论文SSIM: 0.9897\n")
    f.write(f"  实测SSIM: {np.mean(ssim_values):.6f}\n")
    f.write(f"  差异: {0.9897 - np.mean(ssim_values):.6f}\n")

print(f"\n详细结果已保存到 sots_full_results/ 目录")

# 7. 可视化分布
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR分布
    axes[0].hist(psnr_values, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(psnr_values), color='red', linestyle='dashed', linewidth=2,
                    label=f'平均: {np.mean(psnr_values):.2f} dB')
    axes[0].axvline(36.59, color='green', linestyle='dashed', linewidth=2, label='论文: 36.59 dB')
    axes[0].set_xlabel('PSNR (dB)')
    axes[0].set_ylabel('数量')
    axes[0].set_title('PSNR分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # SSIM分布
    axes[1].hist(ssim_values, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(ssim_values), color='red', linestyle='dashed', linewidth=2,
                    label=f'平均: {np.mean(ssim_values):.4f}')
    axes[1].axvline(0.9897, color='green', linestyle='dashed', linewidth=2, label='论文: 0.9897')
    axes[1].set_xlabel('SSIM')
    axes[1].set_ylabel('数量')
    axes[1].set_title('SSIM分布')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sots_full_results/distributions.png', dpi=150)
    print("分布图已保存为 sots_full_results/distributions.png")

except ImportError:
    print("未安装matplotlib，跳过绘图")

# 8. 找出最佳和最差结果
print(f"\n=== 最佳和最差结果 ===")

# 最佳PSNR
best_psnr_idx = np.argmax(psnr_values)
print(f"最佳PSNR: {psnr_values[best_psnr_idx]:.2f} dB - {filenames[best_psnr_idx]}")

# 最差PSNR
worst_psnr_idx = np.argmin(psnr_values)
print(f"最差PSNR: {psnr_values[worst_psnr_idx]:.2f} dB - {filenames[worst_psnr_idx]}")

# 最佳SSIM
best_ssim_idx = np.argmax(ssim_values)
print(f"最佳SSIM: {ssim_values[best_ssim_idx]:.6f} - {filenames[best_ssim_idx]}")

# 最差SSIM
worst_ssim_idx = np.argmin(ssim_values)
print(f"最差SSIM: {ssim_values[worst_ssim_idx]:.6f} - {filenames[worst_ssim_idx]}")

print(f"\n测试完成！")