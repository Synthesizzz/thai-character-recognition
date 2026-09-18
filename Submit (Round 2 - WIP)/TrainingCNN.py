"""
เวอร์ชัน local ของโน้ตบุ๊กที่รันบน Colab (round 2 — ปรับปรุงจาก Submit/ เดิม) — ตัดส่วนที่ผูกกับ Colab ออก
(google.colab.drive, path /content/..., !unzip) แล้วชี้ไปที่ชุดข้อมูลที่แตกไว้แล้ว
ในโฟลเดอร์โปรเจกต์แทน ใช้รันผ่าน VSCode/Terminal บนเครื่องได้โดยตรง:

    python TrainingCNN.py

เปลี่ยนจาก Submit/ รอบแรก (val_acc 98.48%):
- resize 32x32 -> 64x64 (กัน feature map ยุบเหลือ 1x1 ก่อนถึง fc)
- ตัด maxpool ออกใน Net.py (ดู Net.py)
- normalize ด้วย mean/std ของ train set เอง (เดิมแค่หาร 255) -> เซฟลง norm_stats.json ให้ TestingCNN.py ใช้ค่าเดียวกัน
- train_test_split ใส่ stratify + random_state=42 (เดิมสุ่มล้วน 72 คลาส อาจเบี้ยว)
- Adam ใส่ weight_decay=1e-4, เพิ่ม ReduceLROnPlateau scheduler
- CrossEntropyLoss ใส่ class weight แก้ class imbalance
- เพิ่ม dropout ใน Net.py กัน overfit
- n_epochs=60 เป็นเพดานบน + early stopping (patience=10) + save เฉพาะ checkpoint ที่ val_acc ดีที่สุด (model_best.pt)
  แทนที่จะ save โมเดล ณ epoch สุดท้ายเสมอ (เดิม epoch 25 มี val_loss แกว่งขึ้นแล้ว = เริ่ม overfit)
"""
import json
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize, rotate
from skimage.util import random_noise
from tqdm import tqdm

from Net import Net

# ---------- Path ----------
# ชี้ไปที่ "ThaiCharacter Dataset/round2" ที่อยู่ในโฟลเดอร์โปรเจกต์เดียวกัน (สัมพัทธ์กับไฟล์นี้)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "ThaiCharacter Dataset", "round2"))

if not os.path.isdir(DATA_PATH):
    raise FileNotFoundError(
        f"ไม่พบโฟลเดอร์ dataset ที่ {DATA_PATH} — แก้ตัวแปร DATA_PATH ด้านบนให้ตรงกับตำแหน่งจริงบนเครื่อง"
    )

classes = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
print(f"พบ {len(classes)} คลาส")

# ---------- Data Loader + Data Augmentation ----------
IMG_SIZE = 64  # เพิ่มจาก 32 -> 64

def augment_image(img):
    angle = np.random.uniform(-15, 15)  # เทียบเท่า RandomRotation
    img_aug = rotate(img, angle, mode='edge')
    if np.random.rand() < 0.5:
        img_aug = random_noise(img_aug, var=0.005)  # เพิ่ม noise เบาๆ
    return img_aug


MIN_SAMPLES_PER_CLASS = 50

train_img = []
train_label = []

for cls in tqdm(classes, desc="Loading images"):
    cls_folder = os.path.join(DATA_PATH, cls)
    img_files = [f for f in os.listdir(cls_folder) if f.lower().endswith('.jpg')]

    original_imgs = []
    for fname in img_files:
        img = imread(os.path.join(cls_folder, fname), as_gray=True)
        img = resize(img, (IMG_SIZE, IMG_SIZE), preserve_range=True)  # preserve_range=True ป้องกันการหารซ้ำ
        img /= 255.0
        original_imgs.append(img.astype('float32'))

    train_img.extend(original_imgs)
    train_label.extend([cls] * len(original_imgs))

    # เติมภาพด้วย Augmentation เฉพาะคลาสที่ขาด
    n_needed = MIN_SAMPLES_PER_CLASS - len(original_imgs)
    if n_needed > 0:
        for i in range(n_needed):
            base_img = original_imgs[i % len(original_imgs)]
            train_img.append(augment_image(base_img).astype('float32'))
            train_label.append(cls)

train_x = np.array(train_img)
train_y = np.array(train_label)
print("Data shape:", train_x.shape)

# ---------- Training & Validating Set Generation ----------
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
train_y = np.array([class_to_idx[l] for l in train_y])

# stratify=train_y กัน val set สุ่มไม่สมดุลระหว่าง 72 คลาส
train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.2, stratify=train_y, random_state=42
)

# normalize ด้วย mean/std ของ train set เอง (คำนวณจาก train เท่านั้น กัน data leakage)
DATA_MEAN = float(train_x.mean())
DATA_STD = float(train_x.std())
train_x = (train_x - DATA_MEAN) / DATA_STD
val_x = (val_x - DATA_MEAN) / DATA_STD
print(f"mean={DATA_MEAN:.4f}, std={DATA_STD:.4f}")

# เซฟ normalization stats ไว้ให้ TestingCNN.py โหลดไปใช้ตอน inference (ต้อง normalize ด้วยค่าเดียวกันเป๊ะ)
NORM_STATS_PATH = os.path.join(SCRIPT_DIR, 'norm_stats.json')
with open(NORM_STATS_PATH, 'w', encoding='utf-8') as f:
    json.dump({'mean': DATA_MEAN, 'std': DATA_STD, 'img_size': IMG_SIZE}, f)
print(f"บันทึก normalization stats ไว้ที่ {NORM_STATS_PATH}")

train_x = train_x.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
train_x = torch.from_numpy(train_x).to(torch.float32)
train_y = torch.from_numpy(train_y).to(torch.long)

val_x = val_x.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
val_x = torch.from_numpy(val_x).to(torch.float32)
val_y = torch.from_numpy(val_y).to(torch.long)

print("Train:", train_x.shape, train_y.shape)
print("Val:", val_x.shape, val_y.shape)

# ไม่ force เป็น 'cuda' เหมือนตอนอยู่บน Colab เพราะเครื่อง local อาจไม่มี GPU
# -> auto-detect แทน จะได้รันได้ทั้งเครื่องที่มี/ไม่มี GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"ใช้ device: {device}")
if device.type == 'cpu':
    print("คำเตือน: กำลังรันบน CPU การเทรนจะช้ากว่าบน Colab GPU มาก "
          "(ข้อมูล 60,000+ ภาพ กับ ResNet18 อาจใช้เวลาหลายชั่วโมงแทนที่จะเป็นนาที) "
          "ถ้าไม่รีบ แนะนำให้กลับไปเทรนบน Colab แทนแล้วเอา model_best.pt มาทดสอบที่นี่แทน")

# ---------- Model Loader ----------
model = Net(num_classes=len(classes)).to(device)

# ---------- Defining Learning Algorithm ----------
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

class_counts = torch.bincount(train_y, minlength=len(classes)).float()
class_weights = (class_counts.sum() / (len(classes) * class_counts)).to(device)
criterion = CrossEntropyLoss(weight=class_weights).to(device)

# ---------- Training Model ----------
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
n_epochs = 60  # เพดานบน ให้ early stopping ตัดจบเองตาม val_acc จริง
EARLY_STOP_PATIENCE = 10

train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=64)

best_val_acc = 0.0
epochs_no_improve = 0
BEST_MODEL_PATH = os.path.join(SCRIPT_DIR, 'model_best.pt')
LAST_MODEL_PATH = os.path.join(SCRIPT_DIR, 'model_last.pt')

for epoch in tqdm(range(n_epochs), desc="Training"):
    model.train()
    tr_loss, tr_correct, tr_total = 0, 0, 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output_train = model(x_batch)

        loss_train = criterion(output_train, y_batch)
        loss_train.backward()
        optimizer.step()
        tr_loss += loss_train.item()

        predicted = torch.argmax(output_train, dim=1)
        tr_correct += (predicted == y_batch).sum().item()
        tr_total += y_batch.size(0)

    train_losses.append(tr_loss / len(train_loader))
    train_accuracies.append(tr_correct / tr_total)

    # evaluating performance on the validation set
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output_val = model(x_batch)
            val_loss += criterion(output_val, y_batch).item()

            predicted = torch.argmax(output_val, dim=1)
            val_correct += (predicted == y_batch).sum().item()
            val_total += y_batch.size(0)

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_correct / val_total)

    scheduler.step(val_losses[-1])
    cur_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch+1}/{n_epochs} - train_loss: {train_losses[-1]:.4f} - "
          f"train_acc: {train_accuracies[-1]*100:.2f}% - val_loss: {val_losses[-1]:.4f} - "
          f"val_acc: {val_accuracies[-1]*100:.2f}% - lr: {cur_lr:.6f}")

    # save เฉพาะตอน val_acc ดีขึ้น (กันเก็บโมเดลที่ overfit ตอนท้าย ๆ)
    if val_accuracies[-1] > best_val_acc:
        best_val_acc = val_accuracies[-1]
        epochs_no_improve = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping ที่ epoch {epoch+1} (val_acc ไม่ดีขึ้น {EARLY_STOP_PATIENCE} epoch ติดกัน) "
                  f"best_val_acc={best_val_acc*100:.2f}%")
            break

# ---------- Save Model ----------
# model_best.pt ถูก save ไว้ระหว่างเทรนแล้ว (checkpoint ที่ val_acc สูงสุด) ใช้ตัวนั้นสำหรับ inference/submission
torch.save(model.state_dict(), LAST_MODEL_PATH)
print(f"บันทึกโมเดล epoch สุดท้ายไว้ที่ {LAST_MODEL_PATH}")
print(f"best_val_acc ระหว่างเทรน: {best_val_acc*100:.2f}% ({BEST_MODEL_PATH})")
