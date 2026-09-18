# แผนขั้นตอนการทำโปรเจกต์ (Step by Step) พร้อมอ้างอิงสไลด์

> หมายเหตุ: เลขหน้าทั้งหมดตรวจสอบโดยเปิดอ่านจริงทีละหน้าใน `10.pdf` (118 หน้า) ไม่ใช่การประมาณ ส่วน `classification_train.pdf` (4 หน้า) และ `classification_inference.pdf` (2 หน้า) อ่านครบทุกหน้าแล้วเช่นกัน
>
> ไฟล์นี้เป็น **แผน/ลำดับขั้นตอน** เท่านั้น ยังไม่มีการเขียนโค้ดให้ ตามที่ขอไว้

---

## ภาพรวม Workflow (ก่อนเริ่มลงมือ)

**อ้างอิง**: `10.pdf` หน้า 31 — สไลด์ "ขั้นตอนการฝึกสอนเครื่องจักร"

**ทำอะไร**: อ่านผังงาน 7 ขั้นตอนหลัก (Data Loader → Training/Validating Set Generation → Model Loader → Defining Learning Algorithm → Training Model → Evaluating Performance → Is it accepted?)

**ทำไปทำไม**: นี่คือโครงหลักที่ทุกขั้นตอนด้านล่างจะ map เข้ากับ 7 หัวข้อนี้ และตรงกับเกณฑ์ให้คะแนนข้อ "อธิบายถึงขั้นตอนการฝึกสอนแบบจำลอง" พอดี ใช้ผังนี้เป็นสารบัญตอนทำสไลด์นำเสนอได้เลย

---

## ขั้นที่ 1: วางโครงสร้างไฟล์โปรเจกต์

**อ้างอิง**: `10.pdf` หน้า 12 — สไลด์ "หลักการรู้จำภาพ" (แผนภาพ Training Phase / Testing Phase)

**ทำอะไร**: แบ่งโค้ดเป็น 3 ไฟล์ตามธรรมเนียมที่สไลด์แนะนำ
- `Net.py` — กำหนดโครงสร้างโมเดล (CNN class)
- `TrainingCNN.py` — เรียก `Net.py` มาฝึกสอน แล้ว save เป็น `model.pt`
- `TestingCNN.py` — เรียก `Net.py` + โหลด `model.pt` มาทดสอบ/inference

**ทำไปทำไม**: ตรงกับสิ่งที่โจทย์ระบุให้ส่ง ("[ส่งไฟล์โค้ด] – โค้ด Train และ Inference พร้อมไฟล์ Weight") พอดีเป๊ะ ไม่ต้องคิดโครงสร้างเอง

---

## ขั้นที่ 2: สร้าง Label Mapping จากชื่อโฟลเดอร์

**อ้างอิง**: `classification_train.pdf` หน้า 1 — โค้ด `CustomImageDataset` ที่ใช้ `classes` tuple แล้วสร้าง `idx_to_class` / `class_to_idx`

**ทำอะไร**: ใช้ชื่อโฟลเดอร์คลาส (161, 162, ..., 249 ทั้ง 72 โฟลเดอร์ใน `round2/`) เป็น label โดยตรง แทนที่จะหา legend จากภายนอก

**ทำไปทำไม**: แก้ปัญหาที่เจอใน Dataset_Summary.md ว่า "ไม่มีไฟล์ label/legend" — ไม่จำเป็นต้องรู้ว่าคลาส 185 คือตัวอักษรไทยตัวไหนจริงๆ ก็ฝึกโมเดลและวัด accuracy ได้ (แต่ถ้าจะอธิบายในสไลด์นำเสนอว่าคลาสไหนคือตัวอักษรอะไร ควรถามอาจารย์เพิ่ม)

---

## ขั้นที่ 3: สร้าง Custom Dataset Class

**อ้างอิง**:
- `10.pdf` หน้า 97 — โค้ด `CustomImageDataset(Dataset)` เต็มรูปแบบ (`__init__`/`__len__`/`__getitem__`) ใช้ `torchvision.io.read_image`
- `10.pdf` หน้า 99 — ตาราง parameter ของ `DataLoader` (`batch_size`, `shuffle`, `num_workers`)
- `classification_train.pdf` หน้า 1 — เวอร์ชันที่ปรับให้อ่านจากโฟลเดอร์ `img_dir/<class>/<file>` ตรงกับโครงสร้างของเรา

**ทำอะไร**: ปรับ `CustomImageDataset` ให้ชี้ไปที่ `ThaiCharacter Dataset/round2/<class_id>/*.jpg` แทนโฟลเดอร์ตัวอย่างเดิม

**ทำไปทำไม**: โครงสร้างโฟลเดอร์ของเราคือ `round2/<class_id>/*.jpg` ตรงกับ pattern ที่โค้ดตัวอย่างรองรับอยู่แล้ว แทบไม่ต้องเขียนใหม่

---

## ขั้นที่ 4: เลือกกลยุทธ์ Resize ภาพ

**อ้างอิง**: `10.pdf` หน้า 67 — สไลด์ "การปรับภาพก่อนนำเข้าสู่ CNN" (6 กลยุทธ์: Direct Resize, Resize+Padding/Letterbox, Center Crop, Random Resized Crop, ROI Crop, Tiling)

**ทำอะไร**: เลือกใช้ **Resize + Padding (Letterbox)** เป็นหลัก (ไม่ใช่ Direct Resize/Stretch)

**ทำไปทำไม**: ชุดข้อมูลของเรามีภาพขนาดเล็กมากและไม่คงที่ (เฉลี่ย ~16×21 px บางภาพเล็กสุด 2×5 px) การใช้ Direct Resize (บีบ/ยืดภาพให้พอดี) จะทำให้รูปทรงตัวอักษรบิดเบี้ยว ซึ่งสไลด์เตือนไว้ตรงๆ ว่า "วัตถุอาจยืด บีบ หรือรูปร่างผิดเพี้ยน" — Letterbox รักษาสัดส่วนโดยเติมขอบว่างแทน

---

## ขั้นที่ 5: แก้ปัญหา Grayscale vs 3-Channel

**อ้างอิง**: `10.pdf` หน้า 68 — สไลด์ "ข้อควรระวัง" (ภาพตัวอย่าง Lena ขาวดำ → คัดลอกเป็น 3 ช่อง)

**ทำอะไร**: ถ้าจะใช้ Transfer Learning จาก pretrained model ต้อง duplicate ภาพ grayscale ให้เป็น 3 channel (เช่น `image.repeat(3,1,1)` หรือใช้ `Grayscale(num_output_channels=3)`) ก่อนป้อนเข้าโมเดล

**ทำไปทำไม**: ภาพในชุดข้อมูลเราเป็น grayscale (1 channel) ทั้งหมด แต่โมเดล pretrained บน ImageNet ต้องการ input 3 channel (เช่น 227×227×3) — สไลด์หน้านี้เตือนปัญหานี้ไว้ตรงๆ พอดีกับสถานการณ์ของเรา

---

## ขั้นที่ 6: Data Augmentation

**อ้างอิง**:
- `10.pdf` หน้า 102 — โค้ดตัวอย่างเต็ม `v2.Compose([PILToTensor, RandomResizedCrop, RandomHorizontalFlip, ToDtype, Normalize])`
- `10.pdf` หน้า 103-104 — หมวด Geometry: `Resize`, `RandomResize`, `RandomCrop`, `RandomHorizontalFlip`, `RandomZoomOut`, `RandomRotation`, `Pad`
- `10.pdf` หน้า 105 — หมวด Color: `Grayscale`, `RGB`, `GaussianBlur`, `RandomAdjustSharpness` + Composition: `RandomApply`
- `10.pdf` หน้า 106 — หมวด Miscellaneous/Conversion: `Normalize`, `PILToTensor`, `ToDtype`
- `10.pdf` หน้า 107 — Auto-Augmentation: `AugMix`, `CutMix`

**ทำอะไร**: ประกอบ pipeline `v2.Compose([...])` โดยเลือกเทคนิคที่เหมาะกับตัวอักษรลายมือ เช่น `RandomRotation` (มุมน้อยๆ), `RandomAdjustSharpness`, `GaussianBlur` เบาๆ — **หลีกเลี่ยง** `RandomHorizontalFlip` (ตัวอักษรไทยพลิกซ้าย-ขวาแล้วความหมายเปลี่ยน/ผิด) ส่วนคลาสที่มีภาพน้อยมาก (163, 177 มี 1 ภาพ) ให้พิจารณาใช้ `AugMix`/`CutMix` เพื่อสร้างตัวอย่างเพิ่ม

**ทำไปทำไม**: ตอบโจทย์เกณฑ์ให้คะแนนข้อ "มีการใช้งานเทคนิคการสังเคราะห์ข้อมูล (1.5%)" โดยตรง และเป็นสิ่งจำเป็นมาก เพราะ dataset มี class imbalance รุนแรง (บางคลาสมีแค่ 1-3 ภาพ) ถ้าไม่ augment คลาสเหล่านี้จะแบ่ง train/val ไม่ได้เลยและโมเดลจะไม่เห็นตัวอย่างเพียงพอ

---

## ขั้นที่ 7: แบ่ง Train/Validation 80/20

**อ้างอิง**: `10.pdf` หน้า 35 (โค้ด `train_test_split`) และหน้า 32 (import `sklearn.model_selection.train_test_split`)

**ทำอะไร**: ใช้ `train_test_split(X, y, test_size=0.2)` แบ่งข้อมูลแต่ละคลาส

**ทำไปทำไม**: โจทย์กำหนดตรงๆ ว่า "มีการแบ่งชุดข้อมูลออกเป็น Train และ Validation ในสัดส่วน 80% ต่อ 20%" — ข้อควรระวัง: คลาสที่มีภาพ 1 ภาพ (163, 177) จะ split ไม่ได้ตามสัดส่วนนี้ ต้อง augment ก่อน split หรือมีข้อยกเว้นสำหรับคลาสเหล่านี้

---

## ขั้นที่ 8: สร้าง DataLoader

**อ้างอิง**: `10.pdf` หน้า 99 — ตาราง parameter (`batch_size`, `shuffle`, `num_workers`)

**ทำอะไร**: ห่อ Dataset ด้วย `DataLoader(dataset, batch_size=..., shuffle=True)` แยกสำหรับ train/val

**ทำไปทำไม**: จัดกลุ่มข้อมูลเป็น batch ป้อนเข้าโมเดลอย่างมีประสิทธิภาพ, `shuffle=True` เฉพาะ train set (สุ่มลำดับใหม่ทุก epoch)

---

## ขั้นที่ 9: เลือก Backbone สำหรับ Transfer Learning

**อ้างอิง**:
- `10.pdf` หน้า 92 — ตาราง pretrained model เต็ม (VGG, ResNet, SqueezeNet, EfficientNet, DenseNet, MobileNet, GoogLeNet ฯลฯ พร้อม Acc@1/Acc@5/Params/GFLOPS)
- `10.pdf` หน้า 93 — โค้ดตัวอย่าง `from torchvision.models import resnet50, ResNet50_Weights` → `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`

**ทำอะไร**: เลือก backbone ที่เหมาะกับภาพขนาดเล็ก/ทรัพยากรจำกัด เช่น `ResNet18`/`ResNet50` หรือ `EfficientNet_B0` (พารามิเตอร์น้อย, GFLOPS ต่ำ) แล้วแทนที่ layer สุดท้าย (`model.fc = nn.Linear(..., 72)`) ให้ output ตรงกับ 72 คลาส

**ทำไปทำไม**: ตอบโจทย์เกณฑ์ให้คะแนนข้อ "มีการใช้งานเทคนิคการถ่ายโอนความรู้ (1.5%)" — ต้องปรับ input เป็น 3-channel ตามขั้นที่ 5 ก่อนใช้ backbone เหล่านี้

---

## ขั้นที่ 10: กำหนด Loss Function และ Optimizer

**อ้างอิง**: `10.pdf` หน้า 42-43 (Loss Functions for Classification), หน้า 54 (`criterion`/`backward`/`step` อธิบายละเอียด)

**ทำอะไร**: ใช้ `criterion = nn.CrossEntropyLoss()` (เพราะเป็น multi-class single-label 72 คลาส ไม่ใช่ multilabel) และ `optimizer = optim.SGD(model.parameters(), lr=..., momentum=0.9)` หรือ `Adam`

**ทำไปทำไม**: CrossEntropyLoss คู่กับ Softmax คือคู่มาตรฐานสำหรับ multiclass classification (สไลด์อธิบายเปรียบเทียบไว้ว่างานแบบนี้ต่างจาก Multilabel ที่ใช้ Sigmoid+BCEWithLogitsLoss)

---

## ขั้นที่ 11: Training Loop

**อ้างอิง**: `10.pdf` หน้า 45 และ 51 (training loop เต็มรูปแบบ), หรือ `classification_train.pdf` หน้า 3 (เวอร์ชันสั้นกว่า พร้อม `torch.save`)

**ทำอะไร**: วน loop ตาม epoch → `optimizer.zero_grad()` → forward → `loss.backward()` → `optimizer.step()` → เก็บค่า `train_losses`/`val_losses` ไว้ plot

**ทำไปทำไม**: ตอบโจทย์ "อธิบายถึงขั้นตอนการฝึกสอนแบบจำลอง" และเก็บ loss ไว้แสดงกราฟใน slide นำเสนอ

---

## ขั้นที่ 12: Evaluate + คำนวณ Accuracy

**อ้างอิง**: `10.pdf` หน้า 57 (`torch.no_grad()` + `accuracy_score`) หรือหน้า 82 (`model.eval()` + `torch.max()` + correct/total)

**ทำอะไร**: สลับโมเดลเป็น `model.eval()` ใช้ `with torch.no_grad():` แล้วคำนวณ `accuracy_score(val_y, predictions)` หรือ correct/total บน validation set

**ทำไปทำไม**: ตอบโจทย์เกณฑ์ "มีการแสดงประสิทธิภาพของชุดฝึกสอนด้วยค่าความถูกต้อง (Accuracy Rate)" โดยตรง และเป็นตัวเลขหลักที่ใช้ตัดเกรดข้อ "ประสิทธิภาพของแบบจำลอง" (5%) กับภาพทดสอบวันนำเสนอ

---

## ขั้นที่ 13: บันทึกโมเดล (Save Weight)

**อ้างอิง**: `10.pdf` หน้า 55 — `torch.save(model, 'C:/model04.pt')`

**ทำอะไร**: เซฟโมเดลเป็นไฟล์ `.pt` หลังฝึกเสร็จ

**ทำไปทำไม**: โจทย์กำหนดให้ส่ง "ไฟล์ Weight ที่ฝึกสอนเรียบร้อย" คู่กับโค้ด

---

## ขั้นที่ 14: เขียน Inference Script (`TestingCNN.py`)

**อ้างอิง**:
- `classification_inference.pdf` หน้า 1-2 — เทมเพลตเต็ม: โหลด model เปล่า → `load_state_dict` → transform ภาพเดียว → `softmax` → `torch.max` → class + confidence
- `10.pdf` หน้า 62-65 — เวอร์ชัน batch: โหลด `model04.pt` → อ่านภาพจาก CSV → `torch.no_grad()` → `softmax` → `np.argmax` → เขียนผลลง `out.csv`

**ทำอะไร**: เขียนสคริปต์แยกสำหรับทดสอบ/inference โดยไม่ต้องเทรนซ้ำ รองรับทั้งกรณีทดสอบภาพเดี่ยว (วันนำเสนอ) และกรณีทดสอบเป็นชุด (batch)

**ทำไปทำไม**: วันที่ 25 ก.ย. ต้องมีขั้นตอน "ทดสอบกับชุดทดสอบ" ต่อหน้ากรรมการ (13:00-13:45) จึงต้องมีสคริปต์ inference ที่รันได้จริงและเร็ว ไม่ใช่แค่โค้ดเทรน

---

## ขั้นที่ 15: เตรียมสไลด์นำเสนอ

**อ้างอิง**: [Project_info.md](Project_info.md) (เกณฑ์ให้คะแนนข้อ "คะแนนการนำเสนอ 2%") + ผังงาน 7 ขั้นตอนจาก `10.pdf` หน้า 31

**ทำอะไร**: จัดสไลด์ให้ครอบคลุมทุกหัวข้อย่อยที่เกณฑ์ระบุ: แนะนำกลุ่ม, อธิบาย dataset (จำนวนคลาส/ตัวอย่างข้อมูล — ใช้ [Dataset_Summary.md](Dataset_Summary.md)), ความท้าทายของ dataset (class imbalance, ภาพเล็ก), โครงสร้าง CNN โดยรวม, เทคนิค Transfer Learning ที่ใช้, เทคนิค Data Augmentation ที่ใช้, ขั้นตอนการฝึกสอน (ใช้ผังงาน 7 ขั้นตอน), ผล Accuracy บนชุด Train/Val

**ทำไปทำไม**: เกณฑ์ให้คะแนนข้อ "คะแนนการนำเสนอและสื่อประกอบ" มีรายการย่อยเกือบ 10 หัวข้อ การอ้างผังงาน/ตารางจากสไลด์อาจารย์โดยตรงช่วยให้อธิบายตรงประเด็นและไม่ตกหัวข้อ

---

## สรุปตารางอ้างอิงหน้าแบบเร็ว

| ขั้นตอน | ไฟล์ | หน้า |
|---|---|---|
| ผังงาน 7 ขั้นตอน | 10.pdf | 31 |
| โครงสร้าง 3 ไฟล์ | 10.pdf | 12 |
| Label mapping จากโฟลเดอร์ | classification_train.pdf | 1 |
| Custom Dataset class | 10.pdf | 97 |
| DataLoader params | 10.pdf | 99 |
| Resize strategy (6 แบบ) | 10.pdf | 67 |
| Grayscale/3-channel warning | 10.pdf | 68 |
| Augmentation ตัวอย่างเต็ม | 10.pdf | 102 |
| Augmentation - Geometry | 10.pdf | 103-104 |
| Augmentation - Color | 10.pdf | 105 |
| Augmentation - AugMix/CutMix | 10.pdf | 107 |
| Train/Val split 80/20 | 10.pdf | 35 |
| Pretrained model table | 10.pdf | 92 |
| Transfer learning code | 10.pdf | 93 |
| Loss/Optimizer | 10.pdf | 42-43, 54 |
| Training loop | 10.pdf | 45, 51 |
| Evaluate/Accuracy | 10.pdf | 57, 82 |
| Save model | 10.pdf | 55 |
| Inference (ภาพเดี่ยว) | classification_inference.pdf | 1-2 |
| Inference (batch) | 10.pdf | 62-65 |
