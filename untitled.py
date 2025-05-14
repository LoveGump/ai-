import warnings
import os
import copy # 用于深拷贝模型权重

# 忽视警告
warnings.filterwarnings('ignore')

import cv2 # 尽管导入了，但在主训练流程中未使用
from PIL import Image # 尽管导入了，但在主训练流程中未使用
import numpy as np # 尽管导入了，但在主训练流程中未使用
import matplotlib.pyplot as plt # 用于后续可能的绘图，主训练流程中未使用

from tqdm.auto import tqdm # 进度条

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader

# --- 从您的自定义模块导入 ---
# !!! 重要: 确保 torch_py 模块在您的环境中可用 !!!
try:
    from torch_py.Utils import plot_image # 未在主训练流程中使用
    from torch_py.MTCNN.detector import FaceDetector # 未在主训练流程中使用
    from torch_py.MobileNetV1 import MobileNetV1 # 核心模型
    from torch_py.FaceRec import Recognition # 未在主训练流程中使用
except ModuleNotFoundError:
    print("="*50)
    print("错误: 无法导入 torch_py 模块。")
    print("请确保 torch_py 包已正确安装或其路径在 PYTHONPATH 中。")
    print("代码将无法继续执行。")
    print("="*50)
    exit()
# --- 自定义模块导入结束 ---

# ==============================================================================
# 配置参数 (Hyperparameters and Configurations)
# ==============================================================================
# --- 数据相关 ---
DATA_PATH = './datasets/5f680a696ec9b83bb0037081-momodel/data/image' # !!! 修改为您的数据路径 !!!
IMG_HEIGHT = 160 # 图像高度
IMG_WIDTH = 160  # 图像宽度
TEST_SPLIT_RATIO = 0.1 # 验证集划分比例

# --- 训练相关 ---
NUM_CLASSES = 2       # 您的 MobileNetV1 是为2分类任务
BATCH_SIZE = 32       # 批大小
EPOCHS = 100          # 最大训练轮数 (可被早停中断)
LEARNING_RATE = 1e-3  # 初始学习率

# --- 学习率调度器相关 ---
LR_SCHEDULER_PATIENCE = 10 # ReduceLROnPlateau: N轮验证损失未改善则降低学习率
LR_SCHEDULER_FACTOR = 0.2  # ReduceLROnPlateau: 学习率衰减因子

# --- 早停相关 ---
EARLY_STOPPING_PATIENCE = 20 # N轮验证损失未改善则提前停止训练

# --- 模型保存 ---
MODEL_SAVE_DIR = './results' # 模型保存目录
BEST_MODEL_NAME = 'best_mobilenetv1_model.pth' # 最佳模型文件名

# --- 设备配置 ---
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")

# ==============================================================================
# 数据处理函数
# ==============================================================================
def processing_data(data_path, height, width, batch_size, test_split):
    """
    数据处理部分
    :param data_path: 数据路径
    :param height: 高度
    :param width: 宽度
    :param batch_size: 每次读取图片的数量
    :param test_split: 测试集划分比例
    :return: 训练集和验证集的 DataLoader
    """
    # 为训练集和验证集定义不同的转换（通常验证集不做随机增强）
    train_transforms = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(p=0.5), # 50%概率水平翻转
        T.RandomVerticalFlip(p=0.1),   # 10%概率垂直翻转 (如果任务适合)
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05), # 轻微颜色抖动
        T.RandomRotation(degrees=10), # 轻微随机旋转
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # 归一化到[-1, 1]范围 (假设3通道)
        # 如果是单通道图像, 使用: T.Normalize(mean=[0.5], std=[0.5])
        # 如果模型期望ImageNet归一化: T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transforms = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 加载完整数据集
    full_dataset = ImageFolder(data_path) # 先不应用transform，后面划分后再分别应用

    # 划分数据集索引
    train_size = int((1 - test_split) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # 随机划分，确保设置随机种子以复现划分
    torch.manual_seed(42) # 为了可复现的划分
    train_indices, test_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, test_size])

    # 为训练集和测试集创建Subset，并分别应用transforms
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset)

    train_dataset_raw = torch.utils.data.Subset(full_dataset, train_indices.indices)
    test_dataset_raw = torch.utils.data.Subset(full_dataset, test_indices.indices)

    train_dataset = TransformedSubset(train_dataset_raw, transform=train_transforms)
    test_dataset = TransformedSubset(test_dataset_raw, transform=valid_transforms)


    # 创建 DataLoader
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Data loaded: {len(train_dataset)} training images, {len(test_dataset)} validation images.")
    return train_data_loader, valid_data_loader

# ==============================================================================
# 主训练逻辑
# ==============================================================================
def train_model():
    # --- 准备工作 ---
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # 创建模型保存目录

    # --- 加载数据 ---
    print("Loading and processing data...")
    train_loader, valid_loader = processing_data(
        data_path=DATA_PATH,
        height=IMG_HEIGHT,
        width=IMG_WIDTH,
        batch_size=BATCH_SIZE,
        test_split=TEST_SPLIT_RATIO
    )

    # --- 初始化模型、优化器、损失函数、学习率调度器 ---
    print("Initializing model, optimizer, etc.")
    model = MobileNetV1(classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', # 监控验证损失，目标是最小化
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        verbose=True # 当学习率改变时打印信息
    )

    # --- 训练状态变量 ---
    best_val_loss = float('inf')
    epochs_no_improve = 0 # 用于早停计数
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    print(f"Starting training for {EPOCHS} epochs...")
    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        # ** 训练阶段 **
        model.train() # 设置模型为训练模式
        running_train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch")
        for inputs, labels in train_pbar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad() # 清空梯度
            outputs = model(inputs) # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新权重

            running_train_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix(loss=loss.item())
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # ** 验证阶段 **
        model.eval() # 设置模型为评估模式
        running_val_loss = 0.0
        running_val_corrects = 0
        
        val_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]", unit="batch")
        with torch.no_grad(): # 验证时不需要计算梯度
            for inputs, labels in val_pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1) # 获取预测类别

                running_val_loss += loss.item() * inputs.size(0)
                running_val_corrects += torch.sum(preds == labels.data)
                val_pbar.set_postfix(loss=loss.item())

        epoch_val_loss = running_val_loss / len(valid_loader.dataset)
        epoch_val_acc = running_val_corrects.double() / len(valid_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item()) # .item() 转换为Python标量

        print(f"\nEpoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}")

        # --- 学习率调度 ---
        scheduler.step(epoch_val_loss)

        # --- 保存最佳模型 ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, BEST_MODEL_NAME))
            print(f"Epoch {epoch+1}: New best model saved with Val Loss: {best_val_loss:.4f}")
            epochs_no_improve = 0 # 重置早停计数器
        else:
            epochs_no_improve += 1

        # --- 早停检查 ---
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {EARLY_STOPPING_PATIENCE} epochs "
                  f"with no improvement on validation loss.")
            break
            
    print("\nFinished Training.")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    print(f"Best model saved at: {os.path.join(MODEL_SAVE_DIR, BEST_MODEL_NAME)}")
    
    # 可选：绘制训练历史
    plot_training_history(history)


# ==============================================================================
# 辅助函数：绘制训练历史 (可选)
# ==============================================================================
def plot_training_history(history):
    if not history['train_loss']: # 如果没有历史记录 (例如0个epoch)
        print("No training history to plot.")
        return

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    # 损失曲线
    axs[0].plot(history['train_loss'], label='Train Loss')
    axs[0].plot(history['val_loss'], label='Validation Loss')
    axs[0].set_title('Loss Over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # 准确率曲线
    axs[1].plot(history['val_acc'], label='Validation Accuracy', color='green')
    axs[1].set_title('Validation Accuracy Over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)
    
    # 学习率曲线
    axs[2].plot(history['lr'], label='Learning Rate', color='red')
    axs[2].set_title('Learning Rate Over Epochs')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Learning Rate')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plot_save_path = os.path.join(MODEL_SAVE_DIR, 'training_history.png')
    plt.savefig(plot_save_path)
    print(f"Training history plot saved to {plot_save_path}")
    # plt.show() # 如果在Jupyter Notebook等环境中，可以直接显示

# ==============================================================================
# 程序入口
# ==============================================================================
if __name__ == '__main__':
    train_model()