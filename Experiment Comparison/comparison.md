# เปรียบเทียบผลการทดลองทั้งหมด Round 1–5 (สร้างโดย `compare.py` — อย่าแก้มือ)

ไฟล์ข้อมูลเต็ม: `comparison.csv` (ทุกค่า config + ผลลัพธ์ + path เต็ม) · `comparison_history.csv` (train/val ทุก epoch)

## แต่ละ Round ทำอะไรไปบ้าง

### Round 1

เวอร์ชันแรกที่เทรนได้จริงบน Colab: ResNet18 pretrained ทำ Transfer Learning, resize 32x32, หาร 255 อย่างเดียว, แบ่ง train/val แบบสุ่ม, เทรน 25 epoch แล้วเก็บโมเดล epoch สุดท้าย

**ต่างจากรอบก่อน:**

- (รอบแรก)

### Round 2

ปรับ pipeline หลายจุดพร้อมกันเพื่อลด overfit/เพิ่ม generalization: resize 64, ตัด maxpool, dropout, standardize, stratify split, weight decay, LR scheduler, class weight, early stopping, เก็บ best checkpoint (เทรนบน Colab) + แก้บั๊ก preprocess() ที่หาร 255 ซ้ำกับภาพ RGB ตอน inference

**ต่างจากรอบก่อน:**

- resize 32->64
- ตัด maxpool (nn.Identity)
- fc: Linear -> Dropout(0.3)+Linear
- normalize: หาร 255 -> หาร 255 แล้ว (x-mean)/std จาก train set (เก็บ norm_stats.json)
- split: สุ่มล้วน -> stratify + random_state=42
- Adam: เพิ่ม weight_decay=1e-4
- เพิ่ม ReduceLROnPlateau (factor 0.5, patience 3)
- loss: เพิ่ม class weight
- epoch: fix 25 -> เพดาน 60 + early stopping (patience 10)
- เก็บโมเดล: epoch สุดท้าย -> best val_acc
- แก้บั๊ก preprocess() (เช็ค max>1 ก่อนหาร 255)

### Round 3

ทดลอง IMG_SIZE=224 (เท่า resolution ที่ ResNet18 pretrained คุ้นเคย; ภาพจริงเล็กมาก ~18x12 จึงเป็นการ upscale) คืน maxpool, เปลี่ยนเป็น lazy Dataset + cache ภาพต้นฉบับ (แก้ Colab OOM), ย้ายมาเทรน local ด้วย DirectML (AMD RX 7600S)

**ต่างจากรอบก่อน:**

- resize 64->224
- คืน maxpool
- โหลดข้อมูลแบบ lazy Dataset + cache ภาพต้นฉบับใน RAM
- augmentation สุ่มใหม่ทุกครั้งที่ดึงภาพ (ทุก epoch ต่างกัน) หลัง resize
- mean/std ประเมินจาก subset สุ่ม 3,000 ภาพ
- เทรนบน DirectML (RX 7600S) แทน Colab
- DataLoader 4 worker + persistent_workers
- เพิ่มตัวเลือก env (EPOCHS, OPTIMIZER, NUM_WORKERS)

### Round 4

ใช้ pipeline ของ Round 3 ทดลองหลายขนาดภาพ (64/96/128) และเปิด/ปิด maxpool แล้ววัด synthetic ทุกแบบ จากนั้นทำซ้ำ img96_nomp ด้วย seed ต่างกัน 2 รอบเพื่อวัดความแกว่งระหว่างรอบ (ผลแยกที่ runs/<ชื่อ>/)

**ต่างจากรอบก่อน:**

- IMG_SIZE ตั้งได้ด้วย env
- maxpool เปิด/ปิดได้ (NO_MAXPOOL=1) และเก็บค่าใน norm_stats.json
- เพิ่มตัวเลือก SEED (weight เริ่มต้น, shuffle, augmentation) — การแบ่ง train/val ยัง random_state=42
- เก็บผลแยกต่อการทดลอง runs/<ชื่อ>/ (model.pt, norm_stats.json, metrics.json, synthetic.json)
- เพิ่ม evaluate_synthetic.py, run_experiments.py, compare.py

### Round 5

เปลี่ยนเฉพาะ Data Augmentation: สุ่มแปลงภาพเทรน *ทุกภาพ* ระหว่างเทรน (หมุน/ย่อขยาย/เลื่อน, ปรับความหนาเส้น, ลบส่วนของภาพ) เฉพาะชุด train ใช้ข้อมูลของอาจารย์เท่านั้น เทรนที่ 96x96 ไม่มี maxpool บน Colab (ผลเทียบกับ img96_nomp ของ Round 4)

**ต่างจากรอบก่อน:**

- Augmentation ใหม่กับทุกภาพของชุด train (เดิมเฉพาะภาพเติม ~1%): หมุน ±10°/ย่อขยาย 0.9–1.1/เลื่อน ±6% (50%), ปรับความหนาเส้น 1–2% ของขนาดภาพ (30%), ลบส่วนของภาพ 1–2 ก้อน ก้อนละ 10–35% (40%)
- ชุด val ไม่ถูก augment; โมเดล/optimizer/loss/scheduler/early stopping คงเดิม
- เพิ่ม env AUGMENT, AUG_PREVIEW; TestingCNN แก้โหลดน้ำหนักจาก CUDA (map_location='cpu') และภาพ 1-bit (bool->uint8)

## ผลลัพธ์ (ณ best epoch = checkpoint ที่เก็บ) + synthetic

synthetic = `datasets for testing/synthetic_test_set` 432 ภาพ (1 ภาพ ≈ 0.23 จุด — ความต่างไม่กี่ภาพคือ noise)

| round | run | size | maxpool | seed | epochs/cap | best ep | train_loss | train_acc | val_loss | val_acc | นาที/epoch | synthetic | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Round 1 | round1 | 32 | on | - | 25/25 | 25 | 0.0196 | 99.31% | 0.0860 | 98.48% | - | 53.70% (232/432) | `Round 1/` |
| Round 2 | round2 | 64 | off | - | 41/60 | 31 | 0.0008 | 99.95% | 0.0810 | 98.79% | - | 68.75% (297/432) | `Round 2/` |
| Round 3 | round3_img224 | 224 | on | - | 46/60 | 36 | 0.0010 | 99.93% | 0.0860 | 98.54% | 4.3 | 69.91% (302/432) | `Round 3/` |
| Round 4 | img128 | 128 | on | ไม่ตั้ง | 36/60 | 26 | 0.0019 | 99.82% | 0.0771 | 98.50% | 2.0 | 69.91% (302/432) | `Round 4/runs/img128/` |
| Round 4 | img64_nomp | 64 | off | ไม่ตั้ง | 50/60 | 40 | 0.0007 | 99.96% | 0.1105 | 98.46% | 1.7 | 68.52% (296/432) | `Round 4/runs/img64_nomp/` |
| Round 4 | img96 | 96 | on | ไม่ตั้ง | 40/60 | 30 | 0.0015 | 99.88% | 0.1174 | 98.40% | 1.6 | 68.29% (295/432) | `Round 4/runs/img96/` |
| Round 4 | img96_nomp | 96 | off | ไม่ตั้ง | 46/60 | 36 | 0.0012 | 99.92% | 0.0845 | 98.53% | 2.8 | 70.83% (306/432) | `Round 4/runs/img96_nomp/` |
| Round 4 | img96_nomp_s1 | 96 | off | 1 | 32/60 | 22 | 0.0017 | 99.85% | 0.0826 | 98.49% | 3.2 | 68.98% (298/432) | `Round 4/runs/img96_nomp_s1/` |
| Round 4 | img96_nomp_s2 | 96 | off | 2 | 44/60 | 34 | 0.0011 | 99.93% | 0.0975 | 98.45% | 3.1 | 71.06% (307/432) | `Round 4/runs/img96_nomp_s2/` |
| Round 5 | img96_nomp_aug_s1 | 96 | off | 1 | 55/60 | 45 | 0.0146 | 98.55% | 0.0696 | 98.53% | 2.0 | 69.91% (302/432) | `Round 5/runs/img96_nomp_aug_s1/` |

## ผล ณ epoch สุดท้าย (ก่อน early stopping ตัด)

| run | epoch สุดท้าย | train_loss | train_acc | val_loss | val_acc |
|---|---|---|---|---|---|
| round1 | 25 | 0.0196 | 99.31% | 0.0860 | 98.48% |
| round2 | 41 | 0.0006 | 99.96% | 0.0822 | 98.74% |
| round3_img224 | 46 | 0.0009 | 99.96% | 0.0850 | 98.45% |
| img128 | 36 | 0.0007 | 99.95% | 0.0824 | 98.44% |
| img64_nomp | 50 | 0.0006 | 99.97% | 0.1131 | 98.44% |
| img96 | 40 | 0.0009 | 99.94% | 0.1208 | 98.37% |
| img96_nomp | 46 | 0.0009 | 99.96% | 0.0846 | 98.51% |
| img96_nomp_s1 | 32 | 0.0008 | 99.96% | 0.0855 | 98.46% |
| img96_nomp_s2 | 44 | 0.0009 | 99.94% | 0.0931 | 98.43% |
| img96_nomp_aug_s1 | 55 | 0.0133 | 98.62% | 0.0695 | 98.49% |

## img96_nomp ทำซ้ำ 3 รอบ (ต้นฉบับไม่ตั้ง seed, s1 seed=1, s2 seed=2)

| ค่า | เฉลี่ย | SD | ต่ำสุด | สูงสุด |
|---|---|---|---|---|
| best val_acc | 98.49% | 0.04 | 98.45% | 98.53% |
| synthetic | 70.29% | 1.14 | 68.98% | 71.06% |

## ค่า config ที่ใช้ (ต่อ Round)

| ค่า | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
|---|---|---|---|---|---|
| อุปกรณ์เทรน | Google Colab GPU (CUDA) | Google Colab GPU (CUDA) | DirectML: AMD Radeon RX 7600S (local) | DirectML: AMD Radeon RX 7600S (local) | Google Colab GPU (CUDA) |
| โมเดล | ResNet18 pretrained ImageNet (IMAGENET1K_V1); conv1 ใหม่รับ 1 ช่อง (สุ่มค่าเริ่มต้น); fc -> 72 คลาส; มี maxpool; fc = Linear ตรง ๆ (ไม่มี dropout) | ResNet18 pretrained ImageNet (IMAGENET1K_V1); conv1 ใหม่รับ 1 ช่อง (สุ่มค่าเริ่มต้น); fc -> 72 คลาส; ตัด maxpool; fc = Dropout(0.3)+Linear | ResNet18 pretrained ImageNet (IMAGENET1K_V1); conv1 ใหม่รับ 1 ช่อง (สุ่มค่าเริ่มต้น); fc -> 72 คลาส; มี maxpool; fc = Dropout(0.3)+Linear | ResNet18 pretrained ImageNet (IMAGENET1K_V1); conv1 ใหม่รับ 1 ช่อง (สุ่มค่าเริ่มต้น); fc -> 72 คลาส; maxpool ตามแต่ละรัน; fc = Dropout(0.3)+Linear | ResNet18 pretrained ImageNet (IMAGENET1K_V1); conv1 ใหม่รับ 1 ช่อง (สุ่มค่าเริ่มต้น); fc -> 72 คลาส; ไม่มี maxpool (96x96); fc = Dropout(0.3)+Linear |
| dropout | 0.0 | 0.3 | 0.3 | 0.3 | 0.3 |
| normalize | หาร 255 เป็นช่วง 0-1 เท่านั้น (ไม่ standardize) | หาร 255 แล้ว standardize (x-mean)/std | หาร 255 แล้ว standardize (x-mean)/std | หาร 255 แล้ว standardize (x-mean)/std | หาร 255 แล้ว standardize (x-mean)/std |
| mean / std | - | 0.5592 / 0.4385 (จาก train ทั้งชุด) | 0.5569 / 0.4393 (subset 3,000 ภาพ) | ต่อรัน (ดู comparison.csv) | ต่อรัน (ดู comparison.csv) |
| optimizer | Adam | Adam | Adam | Adam | Adam |
| learning rate | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| weight decay | 0.0 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| LR scheduler | ไม่มี | ReduceLROnPlateau(mode=min, factor=0.5, patience=3) | ReduceLROnPlateau(mode=min, factor=0.5, patience=3) | ReduceLROnPlateau(mode=min, factor=0.5, patience=3) | ReduceLROnPlateau(mode=min, factor=0.5, patience=3) |
| loss | CrossEntropyLoss (ไม่ถ่วงน้ำหนัก) | CrossEntropyLoss + class weight | CrossEntropyLoss + class weight | CrossEntropyLoss + class weight | CrossEntropyLoss + class weight |
| batch size | 64 | 64 | 64 | 64 | 64 |
| epoch สูงสุด | 25 | 60 | 60 | 60 | 60 |
| early stopping | ไม่มี | patience 10 (ดู val_acc) | patience 10 (ดู val_acc) | patience 10 (ดู val_acc) | patience 10 (ดู val_acc) |
| การเก็บโมเดล | epoch สุดท้ายเสมอ | best val_acc | best val_acc | best val_acc | best val_acc |
| แบ่ง train/val | 80/20 สุ่มล้วน (ไม่ stratify) | 80/20 stratify | 80/20 stratify | 80/20 stratify | 80/20 stratify |
| seed ตอนแบ่ง | ไม่ตั้ง | 42 | 42 | 42 | 42 |
| seed ตอนเทรน | ไม่ตั้ง | ไม่ตั้ง | ไม่ตั้ง | ต่อรัน (ดู comparison.csv) | ต่อรัน (ดู comparison.csv) |
| จำนวน train / val | 50,653 / 12,664 (รวม 63,317 = 62,707 จริง + 610 augment) | 50,653 / 12,664 | 50,653 / 12,664 | 50,653 / 12,664 | 50,653 / 12,664 |
| augmentation | เติมภาพคลาสที่มี <50 ภาพให้ครบ 50 (หมุนสุ่ม -15..+15° + Gaussian noise var=0.005 โอกาส 50%) | เหมือน Round 1 (เติมให้ครบ 50 ภาพ/คลาส; หมุน ±15° + noise var 0.005 โอกาส 50%) | เติมให้ครบ 50 ภาพ/คลาส (หมุน ±15° + noise var 0.005 โอกาส 50%) | เติมให้ครบ 50 ภาพ/คลาส (หมุน ±15° + noise var 0.005 โอกาส 50%) | ภาพเติม (ครบ 50/คลาส): หมุน ±15° + noise; ทุกภาพ train: หมุน ±10°/ย่อขยาย/เลื่อน 50%, ปรับความหนาเส้น 30%, ลบส่วนของภาพ 40% |
| augmentation สร้างเมื่อไร | สร้างครั้งเดียวก่อนเทรน (ภาพชุดเดิมทุก epoch) | สร้างครั้งเดียวก่อนเทรน (ภาพชุดเดิมทุก epoch) | สุ่มใหม่ทุกครั้งที่ดึงภาพ (หลัง resize) | สุ่มใหม่ทุกครั้งที่ดึงภาพ (หลัง resize) | สุ่มใหม่ทุกครั้งที่ดึงภาพ (เฉพาะชุด train) |
| การโหลดข้อมูล | พรีโหลดทุกภาพเป็น array เดียว | พรีโหลดทุกภาพเป็น array เดียว | lazy Dataset + cache ภาพต้นฉบับใน RAM | lazy Dataset + cache ภาพต้นฉบับใน RAM | lazy Dataset + cache ภาพต้นฉบับใน RAM |
| DataLoader workers | 0 | 0 | 4 | 4 | 2 |
| ขนาดภาพ / maxpool | 32 / on | 64 / off | 224 / on | 64, 96, 128 / ตามรัน (ดู comparison.csv) | 96 / off |

## หมายเหตุ

- **round1**: เก็บโมเดล epoch สุดท้าย (best=last); ค่า train/val จาก log ของ Colab; synthetic วัดจาก model.pt ในโฟลเดอร์นี้
- **round2**: ตัวส่งปัจจุบัน; ค่า train/val จาก log ของ Colab; synthetic วัดจาก model.pt ในโฟลเดอร์นี้
- **round3_img224**: ประวัติทุก epoch จาก terminal log; เวลา/epoch จากแถบ tqdm
- **img96_nomp_aug_s1**:  [Colab GPU; augmentation ใหม่]
