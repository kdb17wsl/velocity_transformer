import torch
import numpy as np
from config import *
from dataset import get_dataloaders
from model import VelocityTransformer


def evaluate(checkpoint_path=None):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据
    _, val_loader = get_dataloaders()

    # 模型
    model = VelocityTransformer().to(device)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"已加载模型: {checkpoint_path}")
        print(f"  Epoch: {ckpt.get('epoch', '?')}, Val Loss: {ckpt.get('val_loss', '?'):.4f}")
    else:
        print(f"警告: 未找到 checkpoint '{checkpoint_path}'，使用随机权重")

    model.eval()

    all_mae = 0.0
    all_count = 0
    acc_5 = 0.0
    acc_10 = 0.0
    acc_20 = 0.0

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            padding_mask = (targets == -100)

            logits = model(inputs, padding_mask=padding_mask)
            preds = logits.argmax(dim=-1)

            valid = targets != -100
            abs_err = (preds[valid].float() - targets[valid].float()).abs()

            all_mae += abs_err.sum().item()
            all_count += valid.sum().item()

            acc_5 += (abs_err <= 5).sum().item()
            acc_10 += (abs_err <= 10).sum().item()
            acc_20 += (abs_err <= 20).sum().item()

            preds_all.extend(preds[valid].cpu().tolist())
            targets_all.extend(targets[valid].cpu().tolist())

    mae = all_mae / all_count
    acc5 = acc_5 / all_count * 100
    acc10 = acc_10 / all_count * 100
    acc20 = acc_20 / all_count * 100

    # 分布统计
    preds_arr = np.array(preds_all)
    targets_arr = np.array(targets_all)

    print(f"\n{'='*50}")
    print(f"评估结果")
    print(f"{'='*50}")
    print(f"总音符数: {all_count}")
    print(f"MAE (平均绝对误差): {mae:.2f}")
    print(f"±5  准确率: {acc5:.1f}%")
    print(f"±10 准确率: {acc10:.1f}%")
    print(f"±20 准确率: {acc20:.1f}%")
    print(f"真实力度均值: {targets_arr.mean():.1f}, 标准差: {targets_arr.std():.1f}")
    print(f"预测力度均值: {preds_arr.mean():.1f}, 标准差: {preds_arr.std():.1f}")
    print(f"相关系数: {np.corrcoef(preds_arr, targets_arr)[0,1]:.4f}")

    # 按力度区间的表现
    print(f"\n按力度区间:")
    for lo, hi in [(0, 32), (33, 64), (65, 96), (97, 127)]:
        mask = (targets_arr >= lo) & (targets_arr <= hi)
        if mask.sum() > 0:
            seg_mae = np.abs(preds_arr[mask].astype(float) - targets_arr[mask].astype(float)).mean()
            seg_acc5 = (np.abs(preds_arr[mask].astype(float) - targets_arr[mask].astype(float)) <= 5).mean() * 100
            print(f"  [{lo:3d}, {hi:3d}]: {mask.sum():6d} 音符, MAE={seg_mae:.2f}, ±5={seg_acc5:.1f}%")

    return mae, acc5


if __name__ == '__main__':
    evaluate()
