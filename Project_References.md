# แหล่งอ้างอิงจาก Learning Slides สำหรับ Project 1 (Thai Character Recognition)

สรุปว่าไฟล์ไหนใน `Learning Slides/` ใช้เป็นแหล่งอ้างอิง/แนวทางทำโปรเจกต์ได้บ้าง เทียบกับข้อกำหนดใน [Project_info.md](Project_info.md) และปัญหาที่พบใน [Dataset_Summary.md](Dataset_Summary.md)

## สรุปภาพรวม: ใช้อะไรก่อน-หลัง

| ลำดับความสำคัญ | ไฟล์ | เหตุผล |
|---|---|---|
| **1 (เทมเพลตโค้ดหลัก)** | `classification_train.pdf`, `classification_inference.pdf` | โค้ด PyTorch ที่รันได้จริงแบบ end-to-end ใกล้เคียงกับโจทย์ที่สุด เอามาปรับใช้ได้เกือบทันที |
| **2 (แหล่งอ้างอิง/อธิบายขั้นตอน)** | `10.pdf` — Practical Implementation of CNN | บทที่ **ตรงตามที่คาดไว้ว่าช่วยได้เยอะ** ครอบคลุมทั้ง workflow, resize strategy, Transfer Learning, Data Augmentation |
| 3 (ข้าม ไม่ต้องอ้างซ้ำ) | `8.pdf` — Pre-trained CNN & Data Augmentation | เนื้อหาซ้ำกับ `10.pdf` เกือบทั้งหมด (โค้ด augmentation/pretrained model ชุดเดียวกัน) ใช้ `10.pdf` แทนพอ |
| ไม่เกี่ยวกับโปรเจกต์นี้ | `Batch Normalization.pdf`, `Lab05.pdf`, `Lab06.pdf` | เป็นแบบฝึกหัดคนละหัวข้อ (คำนวณ similarity มือ, forward prop มือ) ไม่มีโค้ด CNN/PyTorch |

---

## 1. `classification_train.pdf` (4 หน้า) — เทมเพลตสำหรับ Train

โค้ด Jupyter ตัวอย่างจำแนกภาพผลไม้ 10 คลาส โครงสร้างตรงกับสิ่งที่โปรเจกต์นี้ต้องทำ:

- **`CustomImageDataset(Dataset)`** อ่านภาพจากโฟลเดอร์ `img_dir/<class>/<file>` — ตรงกับโครงสร้าง `round2/<class_id>/*.jpg` ของเราเป๊ะ
- มี `classes` tuple → สร้าง `idx_to_class` / `class_to_idx` dict เอง — **แก้ปัญหา "ไม่มีไฟล์ label legend"** ที่ระบุไว้ใน Dataset_Summary.md ได้ทันที (ใช้ชื่อโฟลเดอร์ 161-249 เป็น class label โดยตรง)
- `transforms.Compose([Resize((256,256)), ToImage(), ConvertImageDtype(float32), Normalize(0.5,0.5,0.5)])` — จุดที่ต้องปรับ: ภาพเราเป็น grayscale ขนาดเล็ก ต้องปรับ Normalize เป็น 1 channel หรือ duplicate เป็น 3 channel ถ้าจะใช้ Transfer Learning
- โค้ดแสดงตัวอย่างภาพด้วย `imshow` / `make_grid` — ใช้ตอบโจทย์ข้อ "มีการอธิบายชุดข้อมูล...ตัวอย่างข้อมูลในแต่ละคลาส" ได้เลย
- Transfer Learning: `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)` แล้ว `model.fc = nn.Linear(2048, len(classes))`
- Training loop เต็มรูปแบบ: device handling, `CrossEntropyLoss`, SGD, print loss ระหว่างเทรน, `torch.save(model.state_dict(), 'model.pt')`

## 2. `classification_inference.pdf` (2 หน้า) — เทมเพลตสำหรับ Inference

- โหลด `resnet50()` เปล่า → แทน `.fc` → `load_state_dict(torch.load('model.pt'))`
- Inference ภาพเดี่ยว: ใช้ transform pipeline เดียวกับตอน train → `model.eval()` → `torch.softmax` → `torch.max` ได้ทั้งคลาสที่ทำนายและค่าความมั่นใจ (%)
- นี่คือเทมเพลตพร้อมใช้สำหรับไฟล์ `TestingCNN.py` ที่โจทย์กำหนดให้ส่ง ("[ส่งไฟล์โค้ด] – โค้ด Train และ Inference พร้อมไฟล์ Weight")

## 3. `10.pdf` — Practical Implementation of CNN (118 หน้า) — บทอ้างอิงหลัก

แบ่งเป็น 2 ส่วนใหญ่:

**ส่วนที่ 1 (หน้า ~1-77): พื้นฐาน PyTorch classification workflow**
- Flowchart 7 ขั้นตอน (หน้า ~40): Data Loader → Train/Val Split → Model Loader → Defining Learning Algorithm → Training Model → Evaluating Performance → Is it accepted? — ใช้ตอบข้อ **"อธิบายถึงขั้นตอนการฝึกสอนแบบจำลอง"** ในเกณฑ์ให้คะแนนได้ตรงๆ
- โครงสร้างไฟล์โปรเจกต์แบบ 3 ไฟล์: `Net.py` (นิยามโมเดล) + `TrainingCNN.py` + `TestingCNN.py`, บันทึก/โหลดโมเดลด้วย `torch.save`/`torch.load` — ใช้เป็นโครงสร้างไฟล์ส่งงานได้เลย
- Import ที่ต้องใช้ (หน้า ~61): `sklearn.model_selection.train_test_split` (สำหรับแบ่ง **80/20 train/val** ตามที่โจทย์กำหนด), `sklearn.metrics.accuracy_score`, `skimage.io`, `torch.nn`, `torch.optim`
- โค้ด Data Loader อ่านภาพ+CSV label แบบเต็ม: `imread(path, as_gray=True)`, normalize `/255.0`, reshape เป็น `(N,1,H,W)` — ปรับใช้กับภาพ grayscale ของเราได้ตรงตัว
- Training/Eval loop เต็ม (`optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`, ฝั่ง eval ใช้ `model.eval()` + `torch.no_grad()` + คำนวณ accuracy) — ตอบโจทย์ **"แสดงประสิทธิภาพด้วยค่า Accuracy Rate"**
- ตาราง pretrained backbone (VGG, ResNet, SqueezeNet, EfficientNet ฯลฯ พร้อม ImageNet Acc@1/params/GFLOPS) + โค้ดตัวอย่าง `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)` — ใช้ตอบข้อ **"มีการใช้งานเทคนิคการถ่ายโอนความรู้ (Transfer Learning)"**

**ส่วนที่ 2 (หน้า ~78-118): Custom Dataset/DataLoader + Data Augmentation ← ส่วนที่สำคัญที่สุดสำหรับเรา**
- Custom `Dataset` class เต็มรูปแบบ (หน้า ~96): `__init__`/`__len__`/`__getitem__`
- **Data Augmentation** (หน้า ~101-105) — ตอบโจทย์ **"มีการใช้งานเทคนิคการสังเคราะห์ข้อมูล (1.5%)"** โดยตรง: `torchvision.transforms.v2` เต็มชุด — `Resize`, `RandomResizedCrop`, `RandomCrop`, `RandomHorizontalFlip`, `RandomRotation`, `RandomZoomOut`, `Grayscale`, `GaussianBlur`, `RandomAdjustSharpness`, `Normalize`, และเทคนิคขั้นสูง `AugMix`, `CutMix` พร้อมตัวอย่าง `v2.Compose([...])`
  - **สำคัญมากสำหรับ dataset นี้**: คลาสที่มีภาพแค่ 1-3 ภาพ (163, 177, 204) จำเป็นต้องพึ่ง augmentation หนักๆ (เช่น CutMix/AugMix) ถึงจะพอมีข้อมูลสอนโมเดลได้
- หน้า ~85: **"การปรับภาพก่อนนำเข้าสู่ CNN"** — 6 กลยุทธ์ resize (Direct Resize/Stretch, Resize+Padding/Letterbox, Center Crop, Random Resized Crop, ROI Crop, Tiling) พร้อมข้อดี-ข้อเสีย — ตรงกับปัญหาที่เราเจอว่าภาพมีขนาดเล็กและไม่คงที่ (เฉลี่ย ~16x21 px) มาก **ควรเลือก Resize+Padding (Letterbox) เพื่อไม่ให้ตัวอักษรบิดเบี้ยว**
- หน้า ~87: คำเตือนเรื่อง grayscale vs 3-channel เมื่อใช้ pretrained model ที่รับภาพ RGB ขนาด 227×227×3 — **ตรงกับปัญหาที่ Dataset_Summary.md เตือนไว้พอดี** (ต้อง duplicate channel เป็น 3 channel)

## 4. ไม่เกี่ยวข้อง (ข้ามได้)

- `8.pdf` — เนื้อหาซ้ำกับ `10.pdf` เกือบหมด ไม่ต้องอ้างอิงแยก
- `Batch Normalization.pdf` — สไลด์เดียว พูดเรื่องตำแหน่งวาง BN ก่อน/หลัง activation เป็นความรู้เสริม ไม่ใช่ requirement ของโจทย์
- `Lab05.pdf` / `Lab06.pdf` — Lab05 เป็นแบบฝึกหัดคำนวณ similarity metric ด้วยมือ, Lab06 เป็นแบบฝึกหัด forward propagation ด้วยมือ ไม่มีโค้ด CNN/PyTorch เกี่ยวข้อง

## แผนการใช้งานที่แนะนำ

1. เริ่มจาก `classification_train.pdf` + `classification_inference.pdf` เป็นโครงโค้ดตั้งต้น (`TrainingCNN.py` / `TestingCNN.py`)
2. ปรับ `CustomImageDataset` ให้อ่านจาก `ThaiCharacter Dataset/round2/<class_id>/*.jpg` แทนโครงสร้างผลไม้เดิม และสร้าง label mapping จากชื่อโฟลเดอร์ (161-249)
3. เพิ่ม `train_test_split` แบ่ง 80/20 ตามที่โจทย์กำหนด (อ้างอิงจาก `10.pdf` หน้า ~61)
4. เพิ่ม Data Augmentation ชุด `torchvision.transforms.v2` (อ้างอิงจาก `10.pdf` หน้า ~101-105) โดยเน้นหนักกับคลาสที่มีข้อมูลน้อย
5. เลือกกลยุทธ์ resize แบบ Letterbox (Resize+Padding) ตาม `10.pdf` หน้า ~85 เพื่อรักษารูปทรงตัวอักษร
6. ใช้ Transfer Learning จาก ResNet/VGG/EfficientNet ตาม `10.pdf` โดยต้องแก้ input เป็น 3-channel (duplicate grayscale) ตามคำเตือนหน้า ~87
7. อ้างอิง flowchart 7 ขั้นตอนจาก `10.pdf` หน้า ~40 เวลาทำสไลด์อธิบายขั้นตอนการฝึกสอน
