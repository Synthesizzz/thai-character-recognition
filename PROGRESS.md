# PROGRESS LOG — Project 1: Thai Character & Number Recognition

บันทึกความคืบหน้าการทำโปรเจกต์แบบต่อเนื่อง อัปเดตล่าสุดหลังเทรนโมเดลสำเร็จครั้งแรกบน Colab ใช้ไฟล์นี้เปิดต่อได้เลยถ้าหยุดแล้วกลับมาทำใหม่

ไฟล์ที่เกี่ยวข้อง: [Project_info.md](Project_info.md) (โจทย์/เกณฑ์คะแนน) · [Dataset_Summary.md](Dataset_Summary.md) (สรุป dataset) · [Project_References.md](Project_References.md) (แหล่งอ้างอิงสไลด์) · [Implementation_Plan.md](Implementation_Plan.md) (แผน 15 ขั้นตอนเต็ม) · `Project_1_Thai_Character_&_Number_Recognition_(15_).ipynb` (โน้ตบุ๊กหลักที่รันบน Colab — อัปเดตล่าสุดแล้ว มี normalize/stratify/dropout/scheduler/early stopping) · `Test/` (โฟลเดอร์ inference สำหรับ local, ผลของ round 1) · `Submit/` (ไฟล์ส่งจริง round 1, val_acc 98.48%) · [`Submit (Round 2 - WIP)/`](Submit%20(Round%202%20-%20WIP)/README.md) (โค้ด round 2 ที่ปรับปรุงแล้ว ยังไม่ retrain — ดูรายละเอียดใน README ของโฟลเดอร์นั้น)

**อัปเดต 2026-09-18**: แก้ notebook หลัก 6 จุด (resize 64x64, normalize mean/std, stratify split, ตัด maxpool ออกจาก `Net.py` + เพิ่ม dropout, weight_decay + LR scheduler + class weight, best-checkpoint saving + early stopping) แล้ว port เป็นสคริปต์ไว้ที่ `Submit (Round 2 - WIP)/` — ยังไม่ retrain เพราะเครื่อง local ไม่มี GPU ต้องไปรันบน Colab ต่อ นอกจากนี้จัดโฟลเดอร์ใหม่: ลบ `Local/` ทิ้ง (ซ้ำกับ `Submit/` 100%)

---

## 🎉 ผลการเทรนล่าสุด (รันสำเร็จครั้งแรกบน Colab, 25 epochs)

```
Epoch 25/25 - train_loss: 0.0196 - train_acc: 99.31% - val_loss: 0.0860 - val_acc: 98.48%
```

- **Validation accuracy: 98.48%** (ค่าที่น่าเชื่อถือ เพราะเป็นภาพที่โมเดลไม่เคยเห็นตอนเทรน) — ดีเกินเกณฑ์ขั้นต่ำของโจทย์มาก (ต้องการแค่ >50% ถึงจะได้คะแนน)
- Train accuracy ไต่จาก 91.57% (epoch 1) → 99.31% (epoch 25) อย่างต่อเนื่อง
- Val accuracy ขึ้นไวช่วง 4-5 epoch แรก (97% → 98%+) แล้วแกว่งอยู่ 97.5-98.5% ตลอด — มี **overfitting เล็กน้อย** (val_loss เริ่มขยับขึ้นทีละนิดหลัง epoch 5 ขณะ train_loss ลดต่อเนื่อง) แต่ไม่รุนแรง ไม่จำเป็นต้องแก้ด่วน
- ทดสอบเพิ่มด้วย `TestingCNN.py` กับภาพทั้งโฟลเดอร์คลาส 161 (1,951 ภาพ) → ทายถูก 1,950/1,951 (99.95%) — ตัวเลขนี้สูงกว่าปกติเพราะภาพส่วนใหญ่เคยถูกใช้เทรนไปแล้ว ไม่ใช่ตัวชี้วัดหลัก ให้ยึด val_acc (98.48%) เป็นหลัก

---

## สถานะภาพรวม (เทียบกับผังงาน 7 ขั้นตอน หน้า 31 ของ `10.pdf`)

| Macro-stage | สถานะ |
|---|---|
| 1. Data Loader + Data Augmentation | ✅ เสร็จแล้ว รันจริงผ่าน (63,317 ภาพหลัง augment เติมคลาสขาด) |
| 2. Training & Validating Set Generation | ✅ เสร็จแล้ว (train 50,653 / val 12,664 ภาพ, 80/20) |
| 3. Model Loader (`Net.py`) | ✅ เสร็จแล้ว (ResNet18, conv1=1channel, fc=72 output) |
| 4. Defining Learning Algorithm | ✅ เสร็จแล้ว (Adam lr=0.0001, CrossEntropyLoss, forced cuda) |
| 5. Training Model | ✅ เสร็จแล้ว รันครบ 25 epochs |
| 6. Evaluating Performance | ✅ เสร็จแล้ว (คำนวณ train/val accuracy ทุก epoch) |
| 7. Is it accepted? | ✅ **ผ่าน** — val_acc 98.48% |
| Inference script (`TestingCNN.py`) | ✅ เขียนเสร็จ + ทดสอบกับ `model.pt` จริงแล้ว ที่ `Test/` |

---

## บั๊กที่เจอและแก้ไปแล้วระหว่างทาง (กันลืม/กันเจอซ้ำ)

1. **`.DS_Store` ทำ Data Loader crash** — `os.listdir` ดึงไฟล์ระบบมาปนกับชื่อโฟลเดอร์คลาส → แก้ด้วย `os.path.isdir(...)` filter
2. **Training loop ไม่ได้เทรนจริง** — โค้ด loop เดิมถูกแยกอยู่คนละเซลล์ (3 เซลล์) ทำให้ indent ไม่ซ้อนกันจริง ต้องรวมเป็นเซลล์เดียว
3. **`output_val = model(val_x)` หลุดเข้าไปใน training batch loop** — เรียก validation ทั้งก้อนไม่ใช่ batch และไม่ย้าย device → ลบออก ย้ายไปทำหลัง training loop ของแต่ละ epoch แทน
4. **`class_to_idx` mapping ไม่ครบ** — เดิม map แค่ `train_y` หลัง split, `val_y` เป็นเลขคลาสดิบ (161-249) เกินขอบเขต 72 output → ย้าย mapping ไปทำก่อน split ทั้งก้อน
5. **`Adam(model.parameterrs())` พิมพ์ผิด** → แก้เป็น `model.parameters()`
6. **Data Augmentation เขียนไว้แต่ไม่เคยถูกใช้จริง** — อยู่หลัง Training/Validating Split ในลำดับเดิม ทำให้คำนวณเสร็จแล้วไม่ได้ reassign เข้า `train_x`/`train_y` → ย้ายมารวมเป็นขั้นตอนเดียวกับ Data Loader ตั้งแต่ต้น
7. **`from Net import Net` อยู่ในเซลล์ Import แรกสุด** — แต่ `Net.py` เพิ่งถูกสร้างตอนหลัง (Model Loader) → ลบออกจากเซลล์แรก เหลือ import ตอน Model Loader ที่เดียวพอ
8. **`resize()` หาร 255 ซ้ำสอง** — `skimage.transform.resize()` แปลงเป็นช่วง 0-1 ให้เองอัตโนมัติ แล้วโค้ดไปหาร 255 ซ้ำอีกที → แก้ด้วย `preserve_range=True`
9. **ไม่มีการคำนวณ Accuracy** — เดิม print แค่ loss ไม่พอกับเกณฑ์ "แสดงประสิทธิภาพด้วย Accuracy Rate" → เพิ่ม `train_accuracies`/`val_accuracies` ด้วย `torch.argmax` เทียบ label จริง
10. **Environment**: เจอ `SystemError: bad call flags` ตอน `import torch` (ปัญหา ABI ของ Python 3.13 กับ torch build) และ `Torch not compiled with CUDA enabled` (เพราะไป force-reinstall torch แบบ CPU-only ทับของเดิม) → แก้ด้วย Disconnect and delete runtime แล้วเชื่อมต่อใหม่ ไม่ต้อง reinstall torch เอง (ของ Colab มี CUDA มาให้แล้ว)

---

## การตัดสินใจเรื่องสภาพแวดล้อมการทำงาน

- **เทรนบน Google Colab** (ฟรี GPU T4) — dataset zip อัปโหลดไว้ที่ `/content/drive/MyDrive/Deep Learning/ThaiCharacter Dataset.zip` แล้ว unzip ลง `/content/dataset/round2` ทุกครั้งที่เปิด session ใหม่
- **โครงสร้างไฟล์**: `Net.py` (สร้างผ่าน `%%writefile` ใน Colab) + โน้ตบุ๊กหลักที่รวม Data Loader → Training (เทียบเท่า `TrainingCNN.py`) + โฟลเดอร์ `Test/` (มี `Net.py` สำเนา + `TestingCNN.py`) สำหรับรัน inference บนเครื่อง local โดยไม่ต้องพึ่ง Colab/GPU
- **`Test/TestingCNN.py`**: auto-detect device (ไม่ force cuda เหมือนตอนเทรน เพราะเครื่อง local ไม่มี GPU), ใช้ `map_location=device` ตอน `torch.load` กัน error ข้าม device, hardcode รายชื่อ 72 คลาส (161-249) ไว้ในไฟล์เพราะชุดทดสอบวันนำเสนออาจไม่ใช่โฟลเดอร์เดียวกับตอนเทรน — ทดสอบแล้วใช้งานได้จริง ต้องมี `model.pt` วางในโฟลเดอร์เดียวกันด้วย (ดาวน์โหลดจาก Colab หลังเทรนเสร็จ)

---

## สิ่งที่ยังไม่ได้ทำ / ทำเพิ่มได้ (ไม่เร่งด่วน — โมเดลผ่านเกณฑ์หลักแล้ว)

1. **Resize ยังเป็น Direct Resize** (`resize(img, (32,32))` บีบภาพตรงๆ ไม่รักษาสัดส่วน) — ยังไม่ได้เปลี่ยนเป็น Letterbox ตามที่คุยไว้ตอนแรก แต่ผลลัพธ์ปัจจุบัน (98.48%) ก็ดีอยู่แล้ว ถ้าจะทำเพิ่มเพื่อคะแนนส่วน "เทคนิคที่น่าสนใจ (2%)" ค่อยกลับมาทำ
2. **เทคนิค/แนวคิดที่น่าสนใจเพิ่มเติม (2%)** — ยังไม่มีอะไรแยกออกมาชัดๆ นอกจาก Transfer Learning + Augmentation ที่บังคับอยู่แล้ว (Letterbox ในข้อ 1 อาจใช้ตอบข้อนี้ได้)
3. **สไลด์นำเสนอ (2%)** — ยังไม่เริ่มเลย ต้องครอบคลุม: แนะนำกลุ่ม, อธิบาย dataset (ใช้ Dataset_Summary.md), ความท้าทาย (class imbalance, ภาพเล็ก/ไม่คงที่), โครงสร้าง CNN, Transfer Learning, Data Augmentation, ขั้นตอนฝึกสอน (ใช้ผังงาน 7 ขั้นตอนหน้า 31), ผล accuracy (train 99.31% / val 98.48%)
4. **backup `model.pt` ไว้ใน Google Drive** — กัน Colab session หลุดแล้วต้องเทรนใหม่ (แก้บรรทัด save เป็น path ใน `/content/drive/MyDrive/...`)
5. พิจารณาลด `n_epochs` เหลือ ~15-18 หรือเพิ่ม augmentation ให้หนักขึ้น ถ้าอยากลด overfitting เล็กน้อยที่เห็นใน val_loss ช่วงหลัง — ไม่จำเป็นต้องทำ ผลลัพธ์ปัจจุบันดีเพียงพอแล้ว
