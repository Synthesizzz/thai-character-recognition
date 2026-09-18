# ไฟล์ที่ส่ง (ตาม Project_info.md)

โจทย์กำหนดให้ส่ง: **"[ส่งไฟล์โค้ด] – โค้ด Train และ Inference พร้อมไฟล์ Weight ที่ฝึกสอนเรียบร้อย"**

| สิ่งที่โจทย์ขอ | ไฟล์ | รายละเอียด |
|---|---|---|
| โครงสร้างโมเดล | `Net.py` | นิยาม class `Net` — ResNet18 (Transfer Learning) แก้ `conv1` ให้รับภาพ grayscale 1 channel และแก้ `fc` ให้ output 72 คลาส |
| **โค้ด Train** | `TrainingCNN.py` | Data Loader → Data Augmentation (เติมภาพให้คลาสที่มีข้อมูลน้อย) → แบ่ง Train/Validation 80/20 → Defining Learning Algorithm → Training Model (25 epochs) → Save Model |
| **โค้ด Inference** | `TestingCNN.py` | โหลด `Net.py` + `model.pt` แล้วทำนายภาพเดี่ยวหรือทั้งโฟลเดอร์ (ออกผลเป็น `out.csv` ถ้าเป็นโฟลเดอร์) |
| **ไฟล์ Weight** | `model.pt` | น้ำหนักโมเดลที่เทรนเสร็จแล้วจาก `TrainingCNN.py` บน Google Colab (GPU) |

## ผลการเทรน

```
Epoch 25/25 - train_loss: 0.0196 - train_acc: 99.31% - val_loss: 0.0860 - val_acc: 98.48%
```

**Validation accuracy: 98.48%** (วัดจากข้อมูล 20% ที่โมเดลไม่เคยเห็นตอนเทรน)

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
