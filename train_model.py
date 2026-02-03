import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
from PIL import Image
import copy
import argparse

def font_to_label(font_name):
    font_mapping = {
        'cs': 0,
        'ks': 1,
        'ls': 2,
        'xs': 3,
        'zs': 4,
        # Add more font mappings as needed
    }
    return font_mapping.get(font_name, -1)  # Return -1 for unknown fonts

class CalligraphyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels_char = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        #0 cs; 1  ks; 2 ls; 3 xs; 4 zs
        self.labels_font = []

        supported_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
        char_idx = 0
        for char_name in sorted(os.listdir(data_dir)):
            char_dir = os.path.join(data_dir, char_name)
            if os.path.isdir(char_dir):
                if char_name not in self.char_to_idx:
                    self.char_to_idx[char_name] = char_idx
                    self.idx_to_char[char_idx] = char_name
                    char_idx += 1

                for root, _, files in os.walk(char_dir):
                    for file in files:
                        if file.lower().endswith(supported_exts):
                            self.image_paths.append(os.path.join(root, file))
                            self.labels_font.append(font_to_label(file[0:2]))
                            self.labels_char.append(self.char_to_idx[char_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Skipping corrupted image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        label_char = self.labels_char[idx]
        font_label = self.labels_font[idx]
        if self.transform:
            image = self.transform(image)
        return image, label_char, font_label

class ForkedResNet50(nn.Module):
    def __init__(self, num_chars, num_styles=5):
        super().__init__()

        # Use ImageNet pretrained backbone for better feature transfer
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Shared layers
        self.shared = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3
        )

        # Task-specific layer4
        self.char_layer4 = base.layer4
        self.style_layer4 = copy.deepcopy(base.layer4)

        self.avgpool = base.avgpool

        # Head regularization: small MLPs with dropout to reduce overfitting
        self.dropout = nn.Dropout(0.5)
        self.fc_char = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_chars)
        )
        self.fc_style = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_styles)
        )

    def forward(self, x):
        shared_feat = self.shared(x)

        char_feat = self.char_layer4(shared_feat)
        style_feat = self.style_layer4(shared_feat)

        char_feat = self.avgpool(char_feat)
        style_feat = self.avgpool(style_feat)

        char_feat = torch.flatten(char_feat, 1)
        style_feat = torch.flatten(style_feat, 1)

        char_feat = self.dropout(char_feat)
        style_feat = self.dropout(style_feat)

        char_logits = self.fc_char(char_feat)
        style_logits = self.fc_style(style_feat)

        return char_logits, style_logits


def get_model(num_classes, pretrained=True):
    # `pretrained` flag kept for API compatibility; backbone uses ImageNet weights
    model = ForkedResNet50(num_classes)
    return model

def save_checkpoint(state, filename='checkpoint.pth'):
    """保存训练断点"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None, override_lr=None):
    """
    加载训练断点，支持覆盖学习率。
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        override_lr: 覆盖学习率（可选）。如果设置，加载后会将所有参数组的lr设置为此值
    """
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 覆盖学习率（如果指定）
        if override_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = override_lr
            print(f"Overriding learning rate to {override_lr}")
        
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        loss = checkpoint['loss']
        
        print(f"Resuming training from epoch {start_epoch + 1}, best accuracy: {best_acc:.4f}")
        return start_epoch, best_acc, loss
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting training from scratch.")
        return 0, 0.0, float('inf')

def train_one_epoch(model, dataloader, criterion_char, criterion_font, optimizer, alpha, device, start_batch_idx=0, use_amp=False, scaler=None, scheduler=None, step_scheduler_per_batch=False):
    """修改后的train_one_epoch，支持从指定batch开始"""
    model.train()
    running_loss = 0.0
    correct_predictions_char = 0
    correct_predictions_font = 0
    total_samples = 0
    
    # 跳过前面的批次
    for i, (inputs, labels_char, labels_font) in enumerate(dataloader):
        if i < start_batch_idx:
            continue

        inputs, labels_char, labels_font = inputs.to(device), labels_char.to(device), labels_font.to(device)
        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast():
                outputs_char, outputs_font = model(inputs)
                loss_char = criterion_char(outputs_char, labels_char)
                loss_font = criterion_font(outputs_font, labels_font)
                loss = loss_char + alpha * loss_font

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs_char, outputs_font = model(inputs)
            loss_char = criterion_char(outputs_char, labels_char)
            loss_font = criterion_font(outputs_font, labels_font)
            loss = loss_char + alpha * loss_font
            loss.backward()
            optimizer.step()

        # Step scheduler per batch if using OneCycleLR
        if scheduler is not None and step_scheduler_per_batch:
            try:
                scheduler.step()
            except Exception:
                pass

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs_char, 1)
        correct_predictions_char += torch.sum(preds == labels_char.data)
        _, preds = torch.max(outputs_font, 1)
        correct_predictions_font += torch.sum(preds == labels_font.data)
        total_samples += inputs.size(0)

        if i % 50 == 49:
            print(f"  Batch {i+1}/{len(dataloader)}, Loss_char: {loss_char.item():.4f}, Loss_font: {loss_font.item():.4f}")

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc_char = correct_predictions_char.double() / total_samples if total_samples > 0 else 0.0
    epoch_acc_font = correct_predictions_font.double() / total_samples if total_samples > 0 else 0.0
    epoch_acc = (epoch_acc_char, epoch_acc_font)
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion_char, criterion_font, alpha, device):
    model.eval()
    running_loss = 0.0
    correct_predictions_char = 0
    correct_predictions_font = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels_char, labels_font in dataloader:
            inputs, labels_char, labels_font = inputs.to(device), labels_char.to(device), labels_font.to(device)
            outputs_char, outputs_font = model(inputs)
            loss_char = criterion_char(outputs_char, labels_char)
            loss_font = criterion_font(outputs_font, labels_font)
            loss = loss_char + alpha * loss_font
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs_char, 1)
            correct_predictions_char += torch.sum(preds == labels_char.data)
            _, preds = torch.max(outputs_font, 1)
            correct_predictions_font += torch.sum(preds == labels_font.data)
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc_char = correct_predictions_char.double() / total_samples if total_samples > 0 else 0.0
    epoch_acc_font = correct_predictions_font.double() / total_samples if total_samples > 0 else 0.0
    epoch_acc = (epoch_acc_char, epoch_acc_font)
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description='Train a calligraphy recognition model.')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the chinese_fonts directory.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--override-lr', type=float, default=None, help='Override learning rate when resuming from checkpoint (use this to fix LR to a constant value)')
    parser.add_argument('--checkpoint-freq', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for character loss vs font loss')
    # Use gentler StepLR defaults so LR doesn't collapse on long runs
    parser.add_argument('--scheduler-step', type=int, default=30, help='Decay learning rate every N epochs')
    parser.add_argument('--scheduler-gamma', type=float, default=0.9, help='Learning rate decay factor')
    parser.add_argument('--freeze-backbone-epochs', type=int, default=0, help='Number of epochs to freeze backbone')
    parser.add_argument('--optimizer', choices=['adam', 'adamw'], default='adam', help='Optimizer to use')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing for CrossEntropyLoss')
    parser.add_argument('--use-onecycle', action='store_true', help='Use OneCycleLR scheduler')
    parser.add_argument('--max-lr', type=float, default=0.01, help='max_lr for OneCycleLR')
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision (AMP)')
    parser.add_argument('--use-tensorboard', action='store_true', help='Log metrics to TensorBoard')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Training on CPU will be very slow.")

    # Improved transforms: use RandomResizedCrop for training and consistent val transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    full_dataset = CalligraphyDataset(args.data_dir, transform=data_transforms['train'])
    num_classes = len(full_dataset.char_to_idx)
    print(f"Found {len(full_dataset)} images belonging to {num_classes} character classes and 5 font stlyles.")

    # 检查字符映射文件是否存在，如果不存在则创建
    if not os.path.exists('char_map.json'):
        with open('char_map.json', 'w', encoding='utf-8') as f:
            json.dump(full_dataset.char_to_idx, f, ensure_ascii=False, indent=4)
        print("Character map saved to char_map.json")
    else:
        print("Character map already exists, skipping creation.")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Ensure val uses the validation transforms
    val_dataset.dataset = CalligraphyDataset(args.data_dir, transform=data_transforms['val'])

    # Compute per-class counts on the training subset and build a WeightedRandomSampler
    try:
        train_indices = train_dataset.indices
    except AttributeError:
        # Older PyTorch or fallback
        train_indices = list(range(train_size))

    # Gather labels for train subset
    train_labels = [full_dataset.labels_char[i] for i in train_indices]
    class_sample_counts = {}
    for lbl in train_labels:
        class_sample_counts[lbl] = class_sample_counts.get(lbl, 0) + 1

    # Create weight for each sample inversely proportional to class frequency
    weights = [1.0 / class_sample_counts[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    }

    model = get_model(num_classes).to(device)
    # Use label smoothing to reduce overconfidence
    label_smoothing = getattr(args, 'label_smoothing', 0.0)
    criterion_char = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    criterion_font = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer selection
    optimizer_name = getattr(args, 'optimizer', 'adam')
    weight_decay = getattr(args, 'weight_decay', 1e-4)
    if optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)

    # Scheduler: support OneCycleLR (recommended) or StepLR fallback
    use_onecycle = getattr(args, 'use_onecycle', False)
    if use_onecycle:
        max_lr = getattr(args, 'max_lr', args.lr)
        steps_per_epoch = max(1, len(dataloaders['train']))
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
    else:
        scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    # AMP scaler
    use_amp = getattr(args, 'use_amp', False)
    scaler = GradScaler() if use_amp and torch.cuda.is_available() else None

    # TensorBoard writer
    use_tb = getattr(args, 'use_tensorboard', False)
    writer = SummaryWriter() if use_tb else None

    # 断点续训相关变量
    start_epoch = 0
    best_acc = 0.0
    
    # 如果指定了断点文件，则加载（可选地覆盖学习率）
    if args.resume:
        override_lr = getattr(args, 'override_lr', None)
        start_epoch, best_acc, _ = load_checkpoint(args.resume, model, optimizer, scheduler, override_lr=override_lr)
    
    # Optionally freeze backbone for initial epochs to train heads first
    freeze_backbone_epochs = getattr(args, 'freeze_backbone_epochs', 0)
    if freeze_backbone_epochs > 0:
        for param in model.shared.parameters():
            param.requires_grad = False
        for param in model.char_layer4.parameters():
            param.requires_grad = False
        for param in model.style_layer4.parameters():
            param.requires_grad = False

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print('-' * 10)
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Unfreeze backbone after initial frozen epochs (if any)
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs:
            print("Unfreezing backbone parameters for fine-tuning")
            for param in model.shared.parameters():
                param.requires_grad = True
            for param in model.char_layer4.parameters():
                param.requires_grad = True
            for param in model.style_layer4.parameters():
                param.requires_grad = True

        train_loss, train_acc = train_one_epoch(
            model,
            dataloaders['train'],
            criterion_char,
            criterion_font,
            optimizer,
            args.alpha,
            device,
            use_amp=(scaler is not None),
            scaler=scaler,
            scheduler=(scheduler if use_onecycle else None),
            step_scheduler_per_batch=use_onecycle
        )
        print(f"Train Loss: {train_loss:.4f} Acc_char: {train_acc[0]:.4f} Acc_font: {train_acc[1]:.4f}")

        val_loss, val_acc = evaluate(model, dataloaders['val'], criterion_char, criterion_font, args.alpha, device)
        print(f"Val Loss: {val_loss:.4f} Acc_char: {val_acc[0]:.4f} Acc_font: {val_acc[1]:.4f}")

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc/char_train', float(train_acc[0]), epoch)
            writer.add_scalar('Acc/char_val', float(val_acc[0]), epoch)
            writer.add_scalar('Acc/font_train', float(train_acc[1]), epoch)
            writer.add_scalar('Acc/font_val', float(val_acc[1]), epoch)
            # log learning rate
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # 保存最佳模型
        if val_acc[0] > best_acc:
            best_acc = val_acc[0]
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved to best_model.pth")
        
        # 定期保存断点
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'loss': val_loss,
            }, checkpoint_path)
            
        # 始终保存最新的断点
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'loss': val_loss,
        }, 'latest_checkpoint.pth')
        
        # 更新学习率 (if using epoch-level scheduler)
        if not use_onecycle:
            try:
                scheduler.step()
            except Exception:
                pass

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()