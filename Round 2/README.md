# Round 2 — ปรับ pipeline (Colab, resize 64x64) · **ตัวส่งปัจจุบัน**

> สถานะ: **ตัวส่งปัจจุบัน** (ณ 2026-09-19) · รอบก่อน: [`../Round 1/`](<../Round 1>) · รอบถัดไป: [`../Round 3/`](<../Round 3>)
> รายละเอียดเดิมของโฟลเดอร์นี้ (ตารางไฟล์ที่ส่ง เหตุผลการปรับ บั๊ก `preprocess()`) อยู่ที่ [`README_original.md`](README_original.md)

## ทำอะไรไปบ้าง

ปรับ Round 1 หลายจุดพร้อมกันเพื่อลด overfit และให้ generalize ดีขึ้น แล้วเทรนบน Google Colab (GPU) ใหม่
เทสกับชุด synthetic (ฟอนต์ดิจิทัล 432 ภาพ) ระหว่างทางเจอบั๊ก `preprocess()` ตอน inference แล้วแก้

## ต่างจาก Round 1 ยังไง

| จุด | Round 1 | Round 2 |
|---|---|---|
| ขนาดภาพ | 32x32 | **64x64** |
| `maxpool` ใน `Net.py` | มี | **ตัดออก** (`nn.Identity()`) กัน feature map ยุบเหลือ 1x1 ตอน input เล็ก |
| `fc` | `Linear` | **`Dropout(0.3)` + `Linear`** |
| Normalize | หาร 255 อย่างเดียว | หาร 255 แล้ว **standardize `(x−mean)/std`** จาก train set จริง → เซฟที่ `norm_stats.json` (mean 0.5592, std 0.4385) |
| แบ่ง train/val | สุ่มล้วน | **stratify** + `random_state=42` |
| Optimizer | Adam lr=1e-4 | Adam lr=1e-4 + **weight_decay=1e-4** |
| LR schedule | ไม่มี | **ReduceLROnPlateau** (factor 0.5, patience 3) |
| Loss | CE ธรรมดา | CE + **class weight** (แก้ class imbalance) |
| จำนวน epoch | fix 25 | เพดาน 60 + **early stopping** (patience 10) |
| การเก็บโมเดล | epoch สุดท้ายเสมอ | เก็บเฉพาะ **checkpoint ที่ val_acc ดีที่สุด** |
| บั๊ก `preprocess()` | — | แก้: เช็ค `img.max() > 1.0` ก่อนหาร 255 (ภาพ RGB ถูกหารซ้ำสองครั้ง ทำให้ synthetic เหลือ 1.39% ก่อนแก้) |

การโหลดข้อมูลยังเป็นแบบเดิม (พรีโหลดทุกภาพเข้า array เดียว, augmentation สร้างครั้งเดียวตอนโหลด) seed ไม่ได้ตั้ง

## ผลลัพธ์

| ตัวชี้วัด | ค่า |
|---|---|
| epoch ที่เทรน | 41/60 (early stopping ตัด) · best epoch = 31 |
| ที่ best epoch: train_loss / train_acc | 0.0008 / 99.95% |
| ที่ best epoch: val_loss / **val_acc** | 0.0810 / **98.79%** |
| epoch สุดท้าย (41): val_acc | 98.74% |
| synthetic test (432 ภาพ) | **68.75%** (297/432) — วัดใหม่จาก `model.pt` ในโฟลเดอร์นี้ (Round 1: 53.70%, ดีขึ้น +15 จุด) |

synthetic ต่ำกว่า val มากเพราะเป็นคนละ domain (ชุดเทรนเป็นภาพลายมือ/สแกน ชุดนี้เป็นฟอนต์ดิจิทัล)

ตารางเทียบทุกรอบ: [`../Experiment Comparison/comparison.md`](<../Experiment Comparison/comparison.md>)

## ไฟล์ในโฟลเดอร์

| ไฟล์ | หน้าที่ |
|---|---|
| `Net.py` | ResNet18 + `conv1` 1 ช่อง + ตัด `maxpool` + dropout + `fc` 72 คลาส |
| `TrainingCNN.py` | โค้ด train (resize 64, standardize, stratify, early stopping, best checkpoint) |
| `TestingCNN.py` | inference ภาพเดี่ยว/โฟลเดอร์ — แก้เพิ่ม 2026-09-19: รับ `.jpg/.jpeg/.png/.bmp`, อ่านโฟลเดอร์ย่อยแบบ recursive, พิมพ์ accuracy เมื่อโฟลเดอร์ตั้งชื่อตามคลาส, บังคับ stdout เป็น UTF-8 |
| `model.pt` | น้ำหนัก best checkpoint (epoch 31) |
| `norm_stats.json` | `{"mean", "std", "img_size": 64}` ต้องใช้ค่าเดียวกันตอน inference |
| `README_original.md` | README เดิม |

## วิธีรัน

Inference (รันจากในโฟลเดอร์นี้):
```bash
python TestingCNN.py path/to/image.png
python TestingCNN.py "../datasets for testing/synthetic_test_set"    # -> out.csv + พิมพ์ accuracy
```
Train: แนะนำ Colab (GPU) — `python TrainingCNN.py`
ไลบรารี: `torch`, `torchvision`, `scikit-image`, `scikit-learn`, `pandas`, `numpy`, `tqdm`
