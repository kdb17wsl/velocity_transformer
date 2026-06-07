import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from config import *
from dataset import get_dataloaders
from model import VelocityTransformer


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据
    train_loader, val_loader = get_dataloaders()

    # 模型
    model = VelocityTransformer().to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total_params:.2f}M")

    # 损失函数（-100 是 ignore_index，padding 位置不算 loss）
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 学习率调度（cosine）
    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, total_steps=total_steps,
        pct_start=WARMUP_STEPS / max(total_steps, 1)
    )

    # 混合精度
    scaler = GradScaler() if USE_AMP and device.type == "cuda" else None

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # ========== 训练 ==========
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # padding mask: targets == -100 的位置
            padding_mask = (targets == -100)

            optimizer.zero_grad()

            if scaler is not None:
                with autocast():
                    logits = model(inputs, padding_mask=padding_mask)
                    loss = criterion(logits.view(-1, NUM_CLASSES), targets.view(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs, padding_mask=padding_mask)
                loss = criterion(logits.view(-1, NUM_CLASSES), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        train_time = time.time() - t0

        # ========== 验证 ==========
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_count = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                padding_mask = (targets == -100)

                logits = model(inputs, padding_mask=padding_mask)
                loss = criterion(logits.view(-1, NUM_CLASSES), targets.view(-1))
                val_loss += loss.item()

                # MAE
                preds = logits.argmax(dim=-1)  # (batch, seq)
                valid = targets != -100
                abs_err = (preds[valid].float() - targets[valid].float()).abs().sum()
                val_mae += abs_err.item()
                val_count += valid.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / val_count

        print(f"\nEpoch {epoch:3d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val MAE: {avg_val_mae:.2f} | "
              f"Time: {train_time:.1f}s")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_mae': avg_val_mae,
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
            print(f"  → 已保存最佳模型 (Val Loss: {avg_val_loss:.4f})")

    print(f"\n训练完成！最佳 Val Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    train()
