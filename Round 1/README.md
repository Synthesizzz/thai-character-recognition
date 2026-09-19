# Round 1 — เวอร์ชันแรก (Colab, 25 epochs)

> สถานะ: **archive** ไม่ใช้ส่งแล้ว เก็บไว้เทียบผลย้อนหลัง · รอบถัดไป: [`../Round 2/`](<../Round 2>)
> รายละเอียดเดิมของโฟลเดอร์นี้ (ตารางไฟล์ที่ส่ง วิธีรัน) อยู่ที่ [`README_original.md`](README_original.md)

## ทำอะไรไปบ้าง

รอบแรกสุดที่เทรนได้จริง: ResNet18 pretrained (ImageNet) ทำ Transfer Learning กับภาพตัวอักษร/ตัวเลขไทย 72 คลาส
เทรนบน Google Colab (GPU) ด้วย notebook แล้วแยกเป็นสคริปต์ `TrainingCNN.py` / `TestingCNN.py`

| ส่วน | ค่าที่ใช้ |
|---|---|
| โมเดล | ResNet18 (`IMAGENET1K_V1`), `conv1` ใหม่รับภาพ 1 ช่อง (สุ่มค่าเริ่มต้น), มี `maxpool`, `fc` เป็น `Linear` ตรง ๆ 72 คลาส (ไม่มี dropout) |
| ขนาดภาพ | resize 32x32 |
| Normalize | หาร 255 ให้อยู่ช่วง 0–1 เท่านั้น (ไม่มี standardize ด้วย mean/std) |
| Data Augmentation | เติมภาพให้คลาสที่มีน้อยจนครบ 50 ภาพต่อคลาส (หมุนสุ่ม ±15°, เติม noise 50%) ทำครั้งเดียวตอนโหลดข้อมูล |
| แบ่ง train/val | 80/20 แบบสุ่มล้วน (ไม่ stratify) |
| Optimizer / Loss | Adam lr=1e-4 (ไม่มี weight decay), CrossEntropyLoss ธรรมดา (ไม่มี class weight) |
| จำนวน epoch | fix 25 (ไม่มี early stopping, ไม่มี LR scheduler) |
| การเก็บโมเดล | เก็บโมเดล **epoch สุดท้าย** เสมอ (ไม่เลือก epoch ที่ val ดีที่สุด) |
| Seed | ไม่ได้ตั้ง |

## ต่างจากรอบก่อนยังไง

ไม่มี (เป็นรอบแรก) — บั๊กที่แก้ระหว่างทำ notebook รอบนี้ (ก่อนเทรนสำเร็จ) เช่น `.DS_Store` ทำ data loader crash,
training loop แยกคนละเซลล์, `class_to_idx` map ไม่ครบ, augmentation ถูกคำนวณแล้วไม่ได้ใช้จริง, `resize()` หาร 255 ซ้ำ
ดูรายการเต็มใน `../PROGRESS.md`

## ผลลัพธ์

| ตัวชี้วัด | ค่า |
|---|---|
| epoch ที่เทรน | 25/25 |
| train_loss / train_acc (epoch 25) | 0.0196 / 99.31% |
| val_loss / **val_acc** (epoch 25) | 0.0860 / **98.48%** |
| synthetic test (`datasets for testing/synthetic_test_set`, 432 ภาพ) | **53.70%** (232/432) — วัดใหม่จาก `model.pt` ในโฟลเดอร์นี้ |

สังเกต: val_loss เริ่มสูงขึ้นหลัง epoch ~5 ขณะที่ train_loss ลดต่อเนื่อง (overfit เล็กน้อย) และเพราะเก็บโมเดล
epoch สุดท้าย ผลจึงไม่ใช่ค่า val ที่ดีที่สุดที่เคยเจอระหว่างเทรน

ตารางเทียบทุกรอบ: [`../Experiment Comparison/comparison.md`](<../Experiment Comparison/comparison.md>)

## ไฟล์ในโฟลเดอร์

| ไฟล์ | หน้าที่ |
|---|---|
| `Net.py` | นิยามโมเดล |
| `TrainingCNN.py` | โค้ด train (Data Loader → Augmentation → split → train → save) |
| `TestingCNN.py` | inference ภาพเดี่ยว/ทั้งโฟลเดอร์ (ได้ `out.csv`) |
| `model.pt` | น้ำหนักที่เทรนแล้ว (epoch 25) |
| `README_original.md` | README เดิมของโฟลเดอร์นี้ |

## วิธีรัน

Inference (รันจากในโฟลเดอร์นี้ ต้องมี `model.pt` อยู่ข้างสคริปต์):
```bash
python TestingCNN.py path/to/image.jpg
python TestingCNN.py path/to/test_folder     # -> out.csv
```
Train: แนะนำ Colab (GPU) — `python TrainingCNN.py`
ไลบรารี: `torch`, `torchvision`, `scikit-image`, `scikit-learn`, `pandas`, `numpy`, `tqdm`
