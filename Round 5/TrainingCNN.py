"""
[Round 5] = Round 4 + Data Augmentation แบบสุ่มระหว่างเทรน (on-the-fly) กับ *ทุกภาพของชุด train* (val ไม่ถูก augment)
เปลี่ยนเฉพาะส่วนนี้ ใช้ข้อมูลของอาจารย์ชุดเดิมเท่านั้น: สุ่มหมุน/ย่อขยาย/เลื่อนเล็กน้อย, ปรับความหนาเส้น (บางลง/หนาขึ้น),
ลบส่วนของภาพ (random erasing) — ปิดได้ด้วย env AUGMENT=0 (จะกลับไปเป็นพฤติกรรมเดียวกับ Round 4) · ดูตัวอย่างภาพด้วย AUG_PREVIEW=1
ดู README.md

[Round 4] = โค้ดของ Round 3 + ตัวเลือกผ่าน env: IMG_SIZE, NO_MAXPOOL=1, SEED, RUN_NAME, OPTIMIZER, EPOCHS, NUM_WORKERS
และเก็บผลแต่ละการทดลองแยกที่ runs/<RUN_NAME>/ (model.pt, norm_stats.json, metrics.json) — ดู README.md

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
import sys
import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from skimage.io import imread
from scipy import ndimage as ndi
from skimage.transform import AffineTransform, resize, rotate, warp
from skimage.util import random_noise
from tqdm import tqdm

from Net import Net

# ---------- Path ----------
# ชี้ไปที่ "ThaiCharacter Dataset/round2" ที่อยู่ในโฟลเดอร์โปรเจกต์เดียวกัน (สัมพัทธ์กับไฟล์นี้)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "ThaiCharacter Dataset", "round2"))

IMG_SIZE = int(os.environ.get("IMG_SIZE", 224))  # ขนาด resize (ตั้งด้วย env เช่น IMG_SIZE=96)
USE_MAXPOOL = os.environ.get("NO_MAXPOOL", "0") != "1"  # NO_MAXPOOL=1 = ตัด maxpool ออก (เหมาะกับ input เล็ก)
OPTIMIZER = os.environ.get("OPTIMIZER", "adam").lower()
AUGMENT = os.environ.get("AUGMENT", "1") != "0"  # AUGMENT=0 ปิด augmentation ใหม่ของ Round 5 (= พฤติกรรมเดียวกับ Round 4)
SEED = int(os.environ["SEED"]) if os.environ.get("SEED") else None  # ตั้ง SEED=1 เพื่อให้ผลทำซ้ำได้ (ไม่ตั้ง = สุ่มอิสระเหมือนเดิม)
# ผลของแต่ละการทดลองเก็บแยกโฟลเดอร์ runs/<RUN_NAME>/ (model.pt + norm_stats.json) ไม่เขียนทับกัน
RUN_NAME = os.environ.get("RUN_NAME") or (
    f"img{IMG_SIZE}" + ("" if USE_MAXPOOL else "_nomp") + ("" if OPTIMIZER == "adam" else f"_{OPTIMIZER}")
    + ("_aug" if AUGMENT else "")
)
OUT_DIR = os.path.join(SCRIPT_DIR, "runs", RUN_NAME)
MIN_SAMPLES_PER_CLASS = 50

# ---------- Augmentation ใหม่ (Round 5) — ใช้กับชุด train เท่านั้น ----------
AUG_P_AFFINE = 0.5   # โอกาสสุ่มหมุน (±10°) / ย่อขยาย (0.9–1.1) / เลื่อน (±6% ของขนาดภาพ)
AUG_P_THICK = 0.3    # โอกาสปรับความหนาเส้น: บางลงหรือหนาขึ้น 1–2% ของขนาดภาพ (สุ่มทิศทาง; เกิน 2% เส้นบางจนเหลือเส้นผม)
AUG_P_ERASE = 0.4    # โอกาสลบส่วนของภาพ 1–2 ก้อน (ก้อนละ 10–35% ของความกว้าง/สูง)


def seed_worker(worker_id):
    """DataLoader worker แต่ละตัวมี numpy seed ของตัวเองที่มาจาก seed หลัก (ทำให้ augmentation ทำซ้ำได้เมื่อตั้ง SEED)"""
    np.random.seed((torch.initial_seed() + worker_id) % (2 ** 32))


def augment_image(img):
    angle = np.random.uniform(-15, 15)  # เทียบเท่า RandomRotation
    img_aug = rotate(img, angle, mode='edge')
    if np.random.rand() < 0.5:
        img_aug = random_noise(img_aug, var=0.005)  # เพิ่ม noise เบาๆ
    return img_aug


def augment_train(img, img_size):
    """สุ่มแปลงภาพ (ช่วง 0-1, หมึกเข้มบนพื้นสว่าง) ให้โมเดลทนภาพเอียง/ขยับ/เส้นหนาบางต่างกัน/ตัวอักษรแหว่ง
    เรียกเฉพาะตอนเทรนกับชุด train (ชุด val ไม่เรียก) — สุ่มใหม่ทุกครั้งที่ดึงภาพ"""
    h, w = img.shape
    bg = float(np.percentile(img, 95))  # สีพื้นหลังโดยประมาณ (ไว้ใช้เติมตอนลบ)
    if np.random.rand() < AUG_P_AFFINE:
        a = np.deg2rad(np.random.uniform(-10, 10)); sc = np.random.uniform(0.9, 1.1)
        tx, ty = np.random.uniform(-0.06, 0.06, 2) * img_size; c = img_size / 2.0
        t1 = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]])
        m = np.array([[sc * np.cos(a), -sc * np.sin(a), 0], [sc * np.sin(a), sc * np.cos(a), 0], [0, 0, 1]])
        t2 = np.array([[1, 0, c + tx], [0, 1, c + ty], [0, 0, 1]])
        img = warp(img, AffineTransform(matrix=t2 @ m @ t1).inverse, mode="edge")
    if np.random.rand() < AUG_P_THICK:
        r = max(1, int(round(np.random.uniform(0.01, 0.02) * img_size)))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]; disk = (xx * xx + yy * yy) <= r * r + 1
        # หมึกเข้ม (ค่าต่ำ): maximum_filter = หมึกบางลง, minimum_filter = หมึกหนาขึ้น
        img = ndi.maximum_filter(img, footprint=disk) if np.random.rand() < 0.5 else ndi.minimum_filter(img, footprint=disk)
    if np.random.rand() < AUG_P_ERASE:
        img = img.copy()
        for _ in range(np.random.randint(1, 3)):
            eh = max(1, int(h * np.random.uniform(0.10, 0.35))); ew = max(1, int(w * np.random.uniform(0.10, 0.35)))
            y = np.random.randint(0, h - eh + 1); x = np.random.randint(0, w - ew + 1)
            img[y:y + eh, x:x + ew] = bg
    return np.clip(img, 0.0, 1.0)


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

    def __init__(self, samples, raw_cache, img_size, mean, std, train_aug=False):
        self.train_aug = train_aug  # True = ใช้ augment_train กับทุกภาพ (ชุด train เท่านั้น)
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
        if self.train_aug:
            img = augment_train(img, self.img_size)
        img = (img.astype('float32') - self.mean) / self.std
        tensor = torch.from_numpy(img.astype('float32')).unsqueeze(0)  # (1, H, W)
        return tensor, label


def save_aug_preview(train_samples, raw_cache, classes, out_path, show=("186", "205", "190", "233", "162", "217"), n_aug=7):
    """บันทึกภาพเทียบ: คอลัมน์แรก = ภาพเดิม, ที่เหลือ = augment_train สุ่ม n_aug ครั้ง (ไว้ตรวจว่าตัวอักษรไม่เสียรูป)"""
    from PIL import Image, ImageDraw
    np.random.seed(0)
    tile = 96; pad = 4; rows = [c for c in show if c in classes]
    canvas = Image.new("RGB", (60 + (n_aug + 1) * (tile + pad), 16 + len(rows) * (tile + pad)), "white")
    d = ImageDraw.Draw(canvas); d.text((60, 2), "original | augment_train x%d" % n_aug, fill="black")
    for r, c in enumerate(rows):
        path = next(p for p, aug, lab in train_samples if classes[lab] == c and not aug)
        base = load_resized(raw_cache[path], IMG_SIZE); y = 16 + r * (tile + pad); d.text((4, y + tile // 2), c, fill="black")
        for k in range(n_aug + 1):
            im = base if k == 0 else augment_train(base, IMG_SIZE)
            t = Image.fromarray((np.clip(im, 0, 1) * 255).astype("uint8")).resize((tile, tile), Image.NEAREST)
            canvas.paste(t.convert("RGB"), (60 + k * (tile + pad), y))
    canvas.save(out_path)


def main():
    if SEED is not None:
        # ควบคุม: น้ำหนักเริ่มต้นของ conv1/fc, ลำดับ shuffle, augmentation  (การแบ่ง train/val ตั้ง random_state=42 คงเดิมเสมอ)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
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
    os.makedirs(OUT_DIR, exist_ok=True)
    NORM_STATS_PATH = os.path.join(OUT_DIR, 'norm_stats.json')
    with open(NORM_STATS_PATH, 'w', encoding='utf-8') as f:
        json.dump({'mean': DATA_MEAN, 'std': DATA_STD, 'img_size': IMG_SIZE, 'use_maxpool': USE_MAXPOOL}, f)
    print(f"บันทึก normalization stats ไว้ที่ {NORM_STATS_PATH}")
    print(f"RUN: {RUN_NAME} (IMG_SIZE={IMG_SIZE}, maxpool={'on' if USE_MAXPOOL else 'off'}, optimizer={OPTIMIZER}, augment={'on' if AUGMENT else 'off'})")

    train_dataset = ThaiCharDataset(train_samples, raw_cache, IMG_SIZE, DATA_MEAN, DATA_STD, train_aug=AUGMENT)
    if os.environ.get("AUG_PREVIEW") == "1":  # ดูตัวอย่าง augmentation แล้วออก (ไม่เทรน)
        prev = os.path.join(OUT_DIR, "aug_preview.png")
        save_aug_preview(train_samples, raw_cache, classes, prev)
        print(f"บันทึกตัวอย่าง augmentation ไว้ที่ {prev}")
        return
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
    model = Net(num_classes=len(classes), use_maxpool=USE_MAXPOOL).to(device)

    # ---------- Defining Learning Algorithm ----------
    # เลือก optimizer ด้วย env var: OPTIMIZER=adam (default) หรือ OPTIMIZER=sgd
    # SGD ไม่ใช้ op lerp ที่ DirectML ไม่รองรับ จึงไม่ต้อง fallback ไป CPU
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
    loader_kwargs = dict(batch_size=64, num_workers=num_workers, worker_init_fn=seed_worker)
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    print(f"DataLoader num_workers={num_workers}")

    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    epoch_times = []
    metrics_path = os.path.join(OUT_DIR, 'metrics.json')

    def save_metrics(finished):
        # เก็บผลทุก epoch ไว้เทียบใน compare.py (เขียนทุก epoch เผื่อรันไม่จบ)
        with open(metrics_path, 'w', encoding='utf-8') as mf:
            json.dump({
                'run': RUN_NAME, 'seed': SEED, 'augment': AUGMENT, 'img_size': IMG_SIZE, 'use_maxpool': USE_MAXPOOL, 'optimizer': OPTIMIZER,
                'finished': finished, 'epochs_run': len(train_losses), 'best_val_acc': best_val_acc,
                'best_epoch': best_epoch, 'sec_per_epoch': float(np.mean(epoch_times)) if epoch_times else None,
                'total_sec': float(np.sum(epoch_times)) if epoch_times else 0.0,
                'train_acc': train_accuracies, 'val_acc': val_accuracies,
                'train_loss': train_losses, 'val_loss': val_losses,
            }, mf)

    MODEL_PATH = os.path.join(OUT_DIR, 'model.pt')

    for epoch in tqdm(range(n_epochs), desc="Training"):
        epoch_start = time.time()
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
        epoch_times.append(time.time() - epoch_start)
        cur_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{n_epochs} - train_loss: {train_losses[-1]:.4f} - "
              f"train_acc: {train_accuracies[-1]*100:.2f}% - val_loss: {val_losses[-1]:.4f} - "
              f"val_acc: {val_accuracies[-1]*100:.2f}% - lr: {cur_lr:.6f}")

        save_metrics(False)
        # save เฉพาะตอน val_acc ดีขึ้น (กันเก็บโมเดลที่ overfit ตอนท้าย ๆ)
        if val_accuracies[-1] > best_val_acc:
            best_val_acc = val_accuracies[-1]
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                save_metrics(True)
                print(f"Early stopping ที่ epoch {epoch+1} (val_acc ไม่ดีขึ้น {EARLY_STOP_PATIENCE} epoch ติดกัน) "
                      f"best_val_acc={best_val_acc*100:.2f}%")
                break

    save_metrics(True)
    # ---------- Save Model ----------
    # model.pt ถูก save ไว้ระหว่างเทรนแล้วทุกครั้งที่ val_acc ดีขึ้น (checkpoint ที่ val_acc สูงสุด)
    # ไม่ต้อง save ซ้ำตรงนี้ — เช็คผลลัพธ์สุดท้ายพอ
    print(f"เทรนเสร็จ — best_val_acc: {best_val_acc*100:.2f}% (บันทึกไว้ใน {MODEL_PATH} แล้ว)")


if __name__ == "__main__":
    # กัน UnicodeEncodeError เมื่อ stdout ไม่ใช่ UTF-8 (เช่น redirect ไปไฟล์ / terminal cp1252)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
