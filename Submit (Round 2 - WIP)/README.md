# Round 2 — WIP (ยังไม่ retrain)

โฟลเดอร์นี้คือโค้ดที่ปรับปรุงจาก [`Submit/`](../Submit) (round 1, val_acc 98.48%) ยังไม่ได้เทรนจริง
— **ยังไม่มี `model_best.pt` / `model_last.pt` / `norm_stats.json`** ต้องรัน `TrainingCNN.py` บน Colab
(หรือเครื่องที่มี GPU) ก่อนถึงจะใช้ `TestingCNN.py` ได้

## ต่างจาก Submit/ (round 1) ตรงไหนบ้าง

| จุด | Round 1 (`Submit/`) | Round 2 (ที่นี่) |
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
| Save model | epoch สุดท้ายเสมอ (อาจ overfit ไปแล้ว) | save เฉพาะตอน val_acc ดีขึ้น → `model_best.pt` |

เหตุผลของแต่ละจุด อยู่ในแชทที่คุยกันตอนแก้ (สรุปคร่าว ๆ: round 1 ผลลัพธ์ดีอยู่แล้ว 98.48% แต่มี
overfitting เล็กน้อยให้เห็นใน val_loss หลัง epoch 5 — จุดที่แก้ส่วนใหญ่คือ regularization/robustness
ไม่ใช่แก้เพราะ 98.48% แย่)

## ขั้นตอนถัดไป

1. รัน `TrainingCNN.py` (แนะนำบน Colab GPU เหมือน round 1) — หรือ sync การแก้พวกนี้กลับเข้า notebook
   หลักแล้วรันจาก Colab ตามเดิม จากนั้นค่อยพอร์ตผลมาไว้ที่นี่
2. ดาวน์โหลด `model_best.pt` + `norm_stats.json` มาวางในโฟลเดอร์นี้
3. รัน `TestingCNN.py` sanity-check เหมือนที่ทำกับ round 1 ใน [`Test/`](../Test)
4. เทียบ val_acc ใหม่ กับ 98.48% เดิม ถ้าดีขึ้น/เสถียรขึ้นชัดเจน ค่อยตัดสินใจว่าจะใช้ตัวไหนส่งจริง
   (หรือจะเก็บทั้งสองไว้เทียบในสไลด์นำเสนอก็ได้)

## วิธีรัน (เหมือน Submit/ เดิม)

Train:
```bash
python TrainingCNN.py
```

Inference (ต้องมี `model_best.pt` และ `norm_stats.json` ในโฟลเดอร์นี้ก่อน):
```bash
python TestingCNN.py path/to/image.jpg          # ทดสอบภาพเดียว
python TestingCNN.py path/to/test_folder        # ทดสอบทั้งโฟลเดอร์ -> ได้ out.csv
```
