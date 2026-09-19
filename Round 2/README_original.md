# ไฟล์ที่ส่ง (ตาม Project_info.md)

โจทย์กำหนดให้ส่ง: **"[ส่งไฟล์โค้ด] – โค้ด Train และ Inference พร้อมไฟล์ Weight ที่ฝึกสอนเรียบร้อย"**

| สิ่งที่โจทย์ขอ | ไฟล์ | รายละเอียด |
|---|---|---|
| โครงสร้างโมเดล | `Net.py` | ResNet18 (Transfer Learning) แก้ `conv1` ให้รับ grayscale 1 channel, ตัด `maxpool` ออก (กัน feature map ยุบเร็วเกินไปตอน input 64x64), เพิ่ม `Dropout(0.3)` ก่อน `fc` แก้ output เป็น 72 คลาส |
| **โค้ด Train** | `TrainingCNN.py` | Data Loader → Augmentation → resize 64x64 → normalize ด้วย mean/std ของ train set เอง → split 80/20 (stratify) → Adam + weight_decay + `ReduceLROnPlateau` + class weight → Training พร้อม early stopping (เพดาน 60 epoch, patience=10) → save `model.pt` เฉพาะ checkpoint ที่ val_acc ดีที่สุด |
| **โค้ด Inference** | `TestingCNN.py` | โหลด `Net.py` + `model.pt` + `norm_stats.json` แล้วทำนายภาพเดี่ยวหรือทั้งโฟลเดอร์ (ออกผลเป็น `out.csv` ถ้าเป็นโฟลเดอร์) |
| **ไฟล์ Weight** | `model.pt` | น้ำหนักโมเดลที่เทรนเสร็จแล้วบน Google Colab (GPU) |
| Normalization stats | `norm_stats.json` | `{"mean": 0.5592, "std": 0.4385, "img_size": 64}` คำนวณจาก train set จริง ตอน inference ต้อง normalize ด้วยค่านี้เป๊ะ |

นี่คือ**เวอร์ชันที่ 2** ของโปรเจกต์ (แทนที่เวอร์ชันแรกแล้ว) เวอร์ชันแรกเก็บไว้เป็นหลักฐานที่
[`../Round 1/`](<../Round 1>)

## ผลการเทรน

```
Epoch 41/60 - train_loss: 0.0006 - train_acc: 99.96% - val_loss: 0.0822 - val_acc: 98.74% - lr: 0.000000
Early stopping ที่ epoch 41 (val_acc ไม่ดีขึ้น 10 epoch ติดกัน) best_val_acc=98.79%
```

**Validation accuracy: 98.79%** (วัดจากข้อมูล 20% ที่โมเดลไม่เคยเห็นตอนเทรน, checkpoint จาก epoch ที่
val_acc สูงสุดจริง ไม่ใช่ epoch สุดท้าย)

## ทดสอบกับชุดข้อมูลนอกเหนือจากเทรน (`datasets for testing/synthetic_test_set`, 432 ภาพ, 72 คลาส)

**Accuracy: 68.75% (297/432)** — ต่ำกว่า val_acc เยอะเพราะเป็นคนละ domain กันจริง ๆ (ชุดเทรนเป็นภาพสแกน/
ลายมือ ส่วนชุดนี้เป็นตัวอักษรจากฟอนต์ดิจิทัลสะอาด ๆ) เป็น domain gap ที่รู้ตัว เอาไปพูดเป็น limitation
ของโมเดลในสไลด์ได้ตรงไปตรงมา — เทียบกับเวอร์ชันแรก (53.70%) ดีขึ้น **+15 จุด** บนข้อมูลนอก domain นี้

## เหตุผลที่ปรับจากเวอร์ชันแรก (val_acc 98.48% → 98.79%, synthetic test 53.70% → 68.75%)

| จุด | เวอร์ชันแรก | เวอร์ชันนี้ |
|---|---|---|
| Image resize | 32x32 | 64x64 |
| Net.py maxpool | มี (feature map ยุบเหลือ 1x1 ก่อน fc) | ตัดออก (`nn.Identity()`) |
| Net.py fc | `Linear` ตรง ๆ | `Dropout(0.3)` + `Linear` |
| Normalize | แค่หาร 255 | (x - mean) / std จาก train set จริง → เซฟลง `norm_stats.json` |
| train/val split | สุ่มล้วน | `stratify=train_y, random_state=42` |
| Optimizer | Adam lr=1e-4 | Adam lr=1e-4, `weight_decay=1e-4` |
| LR schedule | ไม่มี | `ReduceLROnPlateau` (factor=0.5, patience=3) |
| Loss | CrossEntropyLoss ธรรมดา | ใส่ class weight แก้ class imbalance |
| Epoch | fix 25 | เพดานบน 60 + early stopping (patience=10) |
| Save model | epoch สุดท้ายเสมอ (อาจ overfit ไปแล้ว) | save เฉพาะตอน val_acc ดีขึ้น |

จุดร่วมกันของทุกข้อคือ **regularization/robustness** ไม่ใช่แก้เพราะเวอร์ชันแรกแย่ (98.48% ก็ผ่านเกณฑ์
สบาย ๆ อยู่แล้ว) แต่ผลจริงพิสูจน์แล้วว่าช่วยเรื่อง generalization จริง ไม่ใช่แค่ปรับ val_acc ให้สวยขึ้นนิดหน่อย

## บั๊กที่เจอระหว่างทดสอบ (แก้แล้ว)

`preprocess()` เดิมหาร 255 ให้ภาพทุกใบแบบไม่มีเงื่อนไข แต่ `imread(as_gray=True)` ของ skimage คืนค่า
ไม่เหมือนกันตามชนิดภาพต้นทาง — ถ้าเป็น RGB จะแปลงผ่าน `rgb2gray` ซึ่งคืน float ช่วง 0-1 มาแล้ว, ถ้าเป็น
grayscale เดี่ยวอยู่แล้ว (แบบชุดเทรนทั้งหมด) จะคืน uint8 ช่วง 0-255 เหมือนเดิม พอไปหาร 255 ซ้ำกับภาพ RGB
ค่าเลยพังเหลือใกล้ 0 ทุกพิกเซล (accuracy บนชุด synthetic ตกไปเหลือ 1.39% ก่อนแก้) แก้โดยเช็ค
`img.max() > 1.0` ก่อนหารใน `preprocess()` — บั๊กนี้ไม่กระทบผลเทรน/val_acc เลย (เพราะ dataset เทรนเป็น
grayscale ล้วน) แต่กระทบ **inference กับภาพ RGB ใด ๆ** เช่นถ่ายด้วยมือถือ สำคัญมากสำหรับตอน demo จริง

## วิธีรัน

Train (แนะนำให้รันบน Google Colab เพื่อใช้ GPU ฟรี — ถ้ารันบนเครื่องที่ไม่มี GPU จะช้ามาก):
```bash
python TrainingCNN.py
```

Inference:
```bash
python TestingCNN.py path/to/image.jpg          # ทดสอบภาพเดียว
python TestingCNN.py path/to/test_folder        # ทดสอบทั้งโฟลเดอร์ -> ได้ out.csv
```

ต้องติดตั้งไลบรารีก่อน: `torch`, `torchvision`, `scikit-image`, `scikit-learn`, `pandas`, `numpy`, `tqdm`
