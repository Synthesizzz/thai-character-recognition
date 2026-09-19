"""
เวอร์ชัน local ของโน้ตบุ๊กที่รันบน Colab (round 3 — ทดลอง IMG_SIZE=224 ต่อจาก Round 2/) —
ตัดส่วนที่ผูกกับ Colab ออก (google.colab.drive, path /content/..., !unzip) แล้วชี้ไปที่ชุดข้อมูลที่แตกไว้แล้ว
ในโฟลเดอร์โปรเจกต์แทน ใช้รันผ่าน VSCode/Terminal บนเครื่องได้โดยตรง:

    python TrainingCNN.py

เปลี่ยนจาก Round 2/ (round 2, val_acc 98.79%, resize 64x64):
- resize 64x64 -> 224x224 (เท่ากับ resolution ที่ ResNet18 pretrained มา) — ภาพต้นทางจริงมีแค่ ~18x12
  พิกเซล เพราะงั้นนี่คือ upscale/interpolate ไม่ได้เพิ่ม detail ใหม่ แค่ทดลองว่า match resolution
  ที่ pretrained weight คุ้นเคยจะช่วยไหม
- เอา maxpool กลับเข้า Net.py (ดู Net.py) — ที่ตัดออกตอน 64x64 เพราะกันยุบเหลือ 1x1 ก่อน fc แต่ 224x224
  ไม่มีปัญหานั้น ถ้าไม่ใส่ maxpool กลับ feature map จะใหญ่เกิน (14x14) กิน compute เปล่าประโยชน์
- โหลดภาพแบบ lazy (custom Dataset) แทนการพรีโหลดทั้งหมดเข้า RAM ก่อนเทรน — ที่ 224x224 การพรีโหลด
  ทั้ง 63,000+ ภาพเป็น float32 array เดียวใช้ RAM ~12GB+ ทำให้ Colab ฟรี OOM ล่มจริงตอนทดสอบ
  Dataset จึงเก็บภาพ "ต้นฉบับขนาดเล็ก" (~18x12 พิกเซล รวมกันแค่ไม่กี่สิบ MB) ไว้ใน RAM ครั้งเดียว
  แล้ว resize/normalize ทีละภาพตอน DataLoader เรียกจริง (peak memory ~1 batch ไม่ว่า resize จะใหญ่แค่ไหน)
- ทำให้เทรนเร็วขึ้น: (1) cache ภาพต้นฉบับ ไม่ต้อง imread จากดิสก์ซ้ำทุก epoch (2) DataLoader ใช้
  num_workers>0 + persistent_workers ซึ่งต้องมี `if __name__ == "__main__":` guard บน Windows จึงย้ายโค้ด
  ทั้งหมดเข้า main() (ตั้งจำนวน worker ได้ด้วย env var NUM_WORKERS)

จุดอื่นเหมือน round 2 ทุกอย่าง (normalize mean/std, stratify split, weight_decay, LR scheduler,
class weight, early stopping, best-checkpoint saving)
"""
import json
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
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

IMG_SIZE = 224  # ทดลอง: ขนาดเท่ากับที่ ResNet18 pretrained มา (จากเดิม 64)
MIN_SAMPLES_PER_CLASS = 50


def augment_image(img):
    angle = np.random.uniform(-15, 15)  # เทียบเท่า RandomRotation
    img_aug = rotate(img, angle, mode='edge')
    if np.random.rand() < 0.5:
        img_aug = random_noise(img_aug, var=0.005)  # เพิ่ม noise เบาๆ
    return img_aug


def load_resized(raw, img_size):
    """ภาพต้นฉบับ (ที่ imread ให้มา) -> resize -> ช่วง 0-1  (ใช้ร่วมกันทั้ง Dataset และการประเมิน mean/std)"""
    img = resize(raw, (img_size, img_size), preserve_range=True)
    # skimage as_gray=True: RGB ต้นทาง -> คืน float 0-1 อยู่แล้ว, grayscale เดี่ยว -> คืน uint8 0-255
    # เช็ค max ก่อนหาร กันหารซ้ำสอง
    if img.max() > 1.0:
        img = img / 255.0
    return img


class ThaiCharDataset(torch.utils.data.Dataset):
    """resize + normalize ทีละภาพตอนถูกเรียก (lazy) จากภาพต้นฉบับขนาดเล็กที่ cache ไว้ใน RAM แล้ว
    (ไม่ต้องอ่านไฟล์จากดิสก์ซ้ำทุก epoch และไม่ต้องเก็บภาพ 224x224 ทั้งชุด)"""

    def __init__(self, samples, raw_cache, img_size, mean, std):
        self.samples = samples      # [(path, do_augment, label)]
        self.raw_cache = raw_cache  # {path: ndarray ต้นฉบับ}
        self.img_size = img_size
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, do_augment, label = self.samples[idx]
        img = load_resized(self.raw_cache[path], self.img_size)
        if do_augment:
            img = augment_image(img)  # random ใหม่ทุกครั้งที่ถูกเรียก = augmentation หลากหลายขึ้นทุก epoch
        img = (img.astype('float32') - self.mean) / self.std
        tensor = torch.from_numpy(img.astype('float32')).unsqueeze(0)  # (1, H, W)
        return tensor, label


def main():
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(
            f"ไม่พบโฟลเดอร์ dataset ที่ {DATA_PATH} — แก้ตัวแปร DATA_PATH ด้านบนให้ตรงกับตำแหน่งจริงบนเครื่อง"
        )

    classes = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    print(f"พบ {len(classes)} คลาส")

    # ---------- Data Loader + Data Augmentation (เก็บ path/label + cache ภาพต้นฉบับขนาดเล็ก) ----------
    samples = []  # (path, augment_flag, label)
    raw_cache = {}

    for cls in tqdm(classes, desc="Indexing images"):
        cls_folder = os.path.join(DATA_PATH, cls)
        img_files = [f for f in os.listdir(cls_folder) if f.lower().endswith('.jpg')]
        img_paths = [os.path.join(cls_folder, f) for f in img_files]

        for p in img_paths:
            samples.append((p, False, cls))
            raw_cache[p] = imread(p, as_gray=True)  # ภาพเล็กมาก (~18x12) รวมทั้งชุดใช้ RAM แค่ไม่กี่สิบ MB

        n_needed = MIN_SAMPLES_PER_CLASS - len(img_paths)
        if n_needed > 0:
            for i in range(n_needed):
                base_path = img_paths[i % len(img_paths)]
                samples.append((base_path, True, cls))

    print(f"รวม {len(samples)} samples ({len(classes)} คลาส)")

    # ---------- Training & Validating Set Generation ----------
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    sample_paths = [s[0] for s in samples]
    sample_aug = [s[1] for s in samples]
    sample_labels = np.array([class_to_idx[s[2]] for s in samples])

    idx_all = np.arange(len(samples))
    train_idx, val_idx = train_test_split(
        idx_all, test_size=0.2, stratify=sample_labels, random_state=42
    )

    train_samples = [(sample_paths[i], sample_aug[i], int(sample_labels[i])) for i in train_idx]
    val_samples = [(sample_paths[i], sample_aug[i], int(sample_labels[i])) for i in val_idx]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # ประเมิน mean/std จาก subset ของ train (ไม่ประมวลผลทั้งชุดที่ 224x224 เพราะเปลือง) -- สุ่มมาสัก 3000 ภาพพอ
    rng = np.random.default_rng(42)
    stats_idx = rng.choice(len(train_samples), size=min(3000, len(train_samples)), replace=False)
    pixel_sum, pixel_sq_sum, pixel_count = 0.0, 0.0, 0
    for i in tqdm(stats_idx, desc="Estimating mean/std"):
        path, _, _ = train_samples[i]
        img = load_resized(raw_cache[path], IMG_SIZE)
        pixel_sum += img.sum()
        pixel_sq_sum += (img ** 2).sum()
        pixel_count += img.size

    DATA_MEAN = float(pixel_sum / pixel_count)
    DATA_STD = float(np.sqrt(pixel_sq_sum / pixel_count - DATA_MEAN ** 2))
    print(f"mean={DATA_MEAN:.4f}, std={DATA_STD:.4f} (ประเมินจาก {len(stats_idx)} ภาพตัวอย่างของ train)")

    # เซฟ normalization stats ไว้ให้ TestingCNN.py โหลดไปใช้ตอน inference (ต้อง normalize ด้วยค่าเดียวกันเป๊ะ)
    NORM_STATS_PATH = os.path.join(SCRIPT_DIR, 'norm_stats.json')
    with open(NORM_STATS_PATH, 'w', encoding='utf-8') as f:
        json.dump({'mean': DATA_MEAN, 'std': DATA_STD, 'img_size': IMG_SIZE}, f)
    print(f"บันทึก normalization stats ไว้ที่ {NORM_STATS_PATH}")

    train_dataset = ThaiCharDataset(train_samples, raw_cache, IMG_SIZE, DATA_MEAN, DATA_STD)
    val_dataset = ThaiCharDataset(val_samples, raw_cache, IMG_SIZE, DATA_MEAN, DATA_STD)

    # auto-detect device: ลอง DirectML ก่อน (GPU AMD/Intel/NVIDIA ผ่าน DirectX 12 บน Windows)
    # แล้วค่อย cuda (NVIDIA) แล้วค่อย cpu เป็น fallback สุดท้าย
    # (import ไว้ในนี้ ไม่ใช่ระดับไฟล์ — worker ของ DataLoader จะได้ไม่ต้อง init DirectML ซ้ำ)
    try:
        import torch_directml
        if torch_directml.device_count() > 0:
            device = torch_directml.device()
            device_name = torch_directml.device_name(0)
        else:
            raise RuntimeError("ไม่พบ DirectML device")
    except (ImportError, RuntimeError):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_name = str(device)

    print(f"ใช้ device: {device_name}")
    if str(device) == 'cpu':
        print("คำเตือน: กำลังรันบน CPU การเทรนจะช้ากว่า GPU มาก "
              "(ข้อมูล 60,000+ ภาพ x 224x224 กับ ResNet18 อาจใช้เวลาหลายชั่วโมงถึงหลายวัน) "
              "ถ้าไม่รีบ แนะนำให้กลับไปเทรนบน Colab แทนแล้วเอา model.pt มาทดสอบที่นี่แทน")

    # ---------- Model Loader ----------
    model = Net(num_classes=len(classes)).to(device)

    # ---------- Defining Learning Algorithm ----------
    # เลือก optimizer ด้วย env var: OPTIMIZER=adam (default) หรือ OPTIMIZER=sgd
    # SGD ไม่ใช้ op lerp ที่ DirectML ไม่รองรับ จึงไม่ต้อง fallback ไป CPU
    OPTIMIZER = os.environ.get("OPTIMIZER", "adam").lower()
    if OPTIMIZER == "sgd":
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)
    elif OPTIMIZER == "adam":
        optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    else:
        raise ValueError(f"OPTIMIZER ต้องเป็น 'adam' หรือ 'sgd' (ได้ '{OPTIMIZER}')")
    print(f"ใช้ optimizer: {OPTIMIZER}")
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_labels_tensor = torch.tensor([s[2] for s in train_samples], dtype=torch.long)
    class_counts = torch.bincount(train_labels_tensor, minlength=len(classes)).float()
    class_weights = (class_counts.sum() / (len(classes) * class_counts)).to(device)
    criterion = CrossEntropyLoss(weight=class_weights).to(device)

    # ---------- Training Model ----------
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    n_epochs = int(os.environ.get("EPOCHS", 60))  # ตั้ง EPOCHS=5 เพื่อทดลองสั้นๆ; เพดานบน ให้ early stopping ตัดจบเองตาม val_acc จริง
    EARLY_STOP_PATIENCE = 10

    # num_workers>0 ใช้ได้เพราะโค้ดทั้งหมดอยู่ใต้ main() + `if __name__ == "__main__":` แล้ว
    # (Windows ใช้ spawn: worker จะ import ไฟล์นี้ใหม่ ถ้าไม่มี guard จะรันเทรนซ้ำใน worker)
    # persistent_workers กัน spawn worker ใหม่ทุก epoch; ตั้ง NUM_WORKERS=0 ถ้าเจอปัญหา
    num_workers = int(os.environ.get("NUM_WORKERS", 4))
    loader_kwargs = dict(batch_size=64, num_workers=num_workers)
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    print(f"DataLoader num_workers={num_workers}")

    best_val_acc = 0.0
    epochs_no_improve = 0
    # adam ยังใช้ model.pt เดิม; sgd แยกเป็น model_sgd.pt เพื่อไม่เขียนทับตอนเทียบผล
    MODEL_PATH = os.path.join(SCRIPT_DIR, 'model.pt' if OPTIMIZER == 'adam' else f'model_{OPTIMIZER}.pt')

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
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping ที่ epoch {epoch+1} (val_acc ไม่ดีขึ้น {EARLY_STOP_PATIENCE} epoch ติดกัน) "
                      f"best_val_acc={best_val_acc*100:.2f}%")
                break

    # ---------- Save Model ----------
    # model.pt ถูก save ไว้ระหว่างเทรนแล้วทุกครั้งที่ val_acc ดีขึ้น (checkpoint ที่ val_acc สูงสุด)
    # ไม่ต้อง save ซ้ำตรงนี้ — เช็คผลลัพธ์สุดท้ายพอ
    print(f"เทรนเสร็จ — best_val_acc: {best_val_acc*100:.2f}% (บันทึกไว้ใน {MODEL_PATH} แล้ว)")


if __name__ == "__main__":
    main()
