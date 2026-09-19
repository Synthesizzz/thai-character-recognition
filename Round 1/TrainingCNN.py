"""
เวอร์ชัน local ของโน้ตบุ๊กที่รันบน Colab — ตัดส่วนที่ผูกกับ Colab ออก
(google.colab.drive, path /content/..., !unzip) แล้วชี้ไปที่ชุดข้อมูลที่แตกไว้แล้ว
ในโฟลเดอร์โปรเจกต์แทน ใช้รันผ่าน VSCode/Terminal บนเครื่องได้โดยตรง:

    python TrainingCNN.py
"""
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
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
        img = resize(img, (32, 32), preserve_range=True)  # preserve_range=True ป้องกันการหารซ้ำ
        # skimage as_gray=True: RGB ต้นทาง -> rgb2gray คืน float 0-1 อยู่แล้ว, grayscale เดี่ยว -> คืน uint8 0-255
        # เช็ค max ก่อนหารกันหารซ้ำสอง (ถ้าหารซ้ำค่าจะพังเหลือใกล้ 0 ทุกพิกเซล)
        if img.max() > 1.0:
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

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)  # แบ่ง 80/20

train_x = train_x.reshape(-1, 1, 32, 32)
train_x = torch.from_numpy(train_x).to(torch.float32)
train_y = torch.from_numpy(train_y).to(torch.long)

val_x = val_x.reshape(-1, 1, 32, 32)
val_x = torch.from_numpy(val_x).to(torch.float32)
val_y = torch.from_numpy(val_y).to(torch.long)

print("Train:", train_x.shape, train_y.shape)
print("Val:", val_x.shape, val_y.shape)

# ---------- Model Loader ----------
model = Net(num_classes=len(classes))

# ---------- Defining Learning Algorithm ----------
optimizer = Adam(model.parameters(), lr=0.0001)
criterion = CrossEntropyLoss()

# ไม่ force เป็น 'cuda' เหมือนตอนอยู่บน Colab เพราะเครื่อง local อาจไม่มี GPU
# -> auto-detect แทน จะได้รันได้ทั้งเครื่องที่มี/ไม่มี GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"ใช้ device: {device}")
if device.type == 'cpu':
    print("คำเตือน: กำลังรันบน CPU การเทรนจะช้ากว่าบน Colab GPU มาก "
          "(ข้อมูล 63,000+ ภาพ กับ ResNet18 อาจใช้เวลาหลายชั่วโมงแทนที่จะเป็นนาที) "
          "ถ้าไม่รีบ แนะนำให้กลับไปเทรนบน Colab แทนแล้วเอา model.pt มาทดสอบที่นี่แทน")

model = model.to(device)
criterion = criterion.to(device)

# ---------- Training Model ----------
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
n_epochs = 25

train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=64)

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

    print(f"Epoch {epoch+1}/{n_epochs} - train_loss: {train_losses[-1]:.4f} - "
          f"train_acc: {train_accuracies[-1]*100:.2f}% - val_loss: {val_losses[-1]:.4f} - "
          f"val_acc: {val_accuracies[-1]*100:.2f}%")

# ---------- Save Model ----------
MODEL_OUT = os.path.join(SCRIPT_DIR, 'model.pt')
torch.save(model.state_dict(), MODEL_OUT)
print(f"บันทึกโมเดลไว้ที่ {MODEL_OUT}")
