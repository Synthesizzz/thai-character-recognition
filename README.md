# Thai Character & Number Recognition

โปรเจกต์วิชา Deep Learning in Medical Image and Video Analysis — พัฒนาแบบจำลอง **CNN + Transfer Learning + Data Augmentation**
สำหรับรู้จำตัวอักษรและตัวเลขภาษาไทย **72 คลาส** (โฟลเดอร์คลาสรหัส 161–249) จากภาพเกรย์สเกลขนาดเล็ก

> กำหนดส่ง/ทดสอบ: **25 ก.ย. 2569 (2026)** — ทดสอบกับชุดทดสอบ 13:00–13:45 · นำเสนอ 13:45–16:30 (กลุ่มละ 10 นาที + ถามตอบ 5 นาที) · ส่งงานก่อน 16:00

## สัญลักษณ์ในตาราง

| สัญลักษณ์ | ความหมาย |
|---|---|
| ✅ | ทำเสร็จแล้ว |
| 🟡 | ทำบางส่วน / มีข้อมูลรองรับแล้วแต่ยังไม่ครบ |
| ⬜ | ยังไม่ได้ทำ |
| ❓ | ต้องรอวันทดสอบจริง (ยังไม่มีผล) |

## ✅ เช็คลิสต์เทียบเกณฑ์การให้คะแนน

### 1) ส่วนคะแนนหลัก

| เกณฑ์ | คะแนน | สถานะ | หลักฐาน / หมายเหตุ |
|---|---|---|---|
| แบบจำลองทำนายภาพทดสอบได้ (>50% ได้ 1% ถึง 100% ได้ 5%) | 5% | ✅ โมเดลพร้อม · ❓ คะแนนจริง | เทรนเสร็จ val_acc **98.79%** (Round 2), ชุด synthetic นอก domain (432 ภาพ) ได้ **68.75%** — เกิน 50% ทั้งสองชุด · ผลบนชุดทดสอบของอาจารย์รอวันสอบ |
| คะแนนจัดลำดับประสิทธิภาพของแบบจำลอง | 3% | ❓ | ขึ้นกับผลวันทดสอบเทียบกับกลุ่มอื่น |
| **เทคนิค:** Transfer Learning | 1.5% | ✅ | ResNet18 pretrained (ImageNet `IMAGENET1K_V1`) แก้ `conv1` เป็น 1 ช่อง + `fc` 72 คลาส — [`Round 2/Net.py`](Round%202/Net.py) |
| **เทคนิค:** Data Augmentation | 1.5% | ✅ | เติมคลาสที่มีน้อยให้ครบ 50 ภาพ/คลาส (หมุนสุ่ม ±15° + noise) — ดู `TrainingCNN.py` ของทุก Round |
| **เทคนิค:** เทคนิค/แนวคิดที่น่าสนใจ | 2% | 🟡 | ทำแล้ว: standardize, class weight, LR scheduler, early stopping + best checkpoint, ทดลองเชิงระบบ 4 รอบ (ขนาดภาพ / maxpool / seed) พร้อมตารางเปรียบเทียบ, วัดกับชุดนอก domain · ยังไม่ได้ทำ: letterbox resize, ablation แบบไม่ใช้ pretrain |
| การนำเสนอและสื่อประกอบ | 2% | ⬜ | ยังไม่ได้ทำสไลด์ (ดูตารางที่ 3) |

### 2) สิ่งที่ต้องส่ง

| รายการ | สถานะ | หมายเหตุ |
|---|---|---|
| โค้ด Train | ✅ | [`Round 2/TrainingCNN.py`](Round%202/TrainingCNN.py) (ตัวส่งปัจจุบัน) |
| โค้ด Inference | ✅ | [`Round 2/TestingCNN.py`](Round%202/TestingCNN.py) — รับภาพเดี่ยวหรือโฟลเดอร์ (`.jpg/.jpeg/.png/.bmp`), ได้ `out.csv` |
| ไฟล์ Weight ที่ฝึกสอนแล้ว | ✅ | `Round 2/model.pt` + `Round 2/norm_stats.json` (ต้องใช้คู่กัน) |
| เอกสารประกอบการนำเสนอ | ⬜ | ยังไม่ได้ทำ |
| ส่งภายใน 25 ก.ย. ก่อน 16:00 | ❓ | รอส่ง |

### 3) เกณฑ์การนำเสนอ (แยก "งานจริง" กับ "อยู่ในสไลด์")

| หัวข้อที่ต้องมีในการนำเสนอ | งานจริงในโปรเจกต์ | อยู่ในสไลด์ | หลักฐาน |
|---|---|---|---|
| ความสวยงามของเอกสาร | — | ⬜ | |
| อธิบายชุดข้อมูล (จำนวนคลาส, จำนวนต่อคลาส, ตัวอย่างต่อคลาส) | ✅ วิเคราะห์แล้ว (72 คลาส / 62,707 ภาพ) · 🟡 ยังไม่ได้เลือกภาพตัวอย่างต่อคลาส | ⬜ | หัวข้อ "ชุดข้อมูล" ด้านล่าง |
| วิเคราะห์ความท้าทายของชุดข้อมูล | ✅ | ⬜ | class imbalance (1 ถึง 5,025 ภาพ/คลาส), ภาพเล็กมากและขนาดไม่คงที่, domain gap กับชุดสังเคราะห์ |
| อธิบายโครงสร้าง CNN ในภาพรวม | ✅ | ⬜ | ResNet18 ปรับ `conv1`/`fc` — `Round 2/Net.py` |
| อธิบาย Transfer Learning | ✅ | ⬜ | ใช้ weight ImageNet เป็นจุดเริ่มต้น เทรนต่อทั้งโมเดล |
| อธิบาย Data Augmentation | ✅ | ⬜ | หมุน ±15° + noise เติมให้ครบ 50 ภาพ/คลาส |
| อธิบายเทคนิค/แนวคิดที่น่าสนใจ | 🟡 | ⬜ | ดูแถว "เทคนิคที่น่าสนใจ" ในตารางที่ 1 |
| อธิบายการทำงานของ CNN เน้นจุดเด่นสถาปัตยกรรม | ⬜ ยังไม่ได้เตรียม | ⬜ | |
| อธิบายเทคนิคที่ใช้งาน | ✅ | ⬜ | ตาราง "เทคนิคที่ใช้" ด้านล่าง |
| อธิบายขั้นตอนการฝึกสอนแบบจำลอง | ✅ | ⬜ | Data Loader → Augmentation → split 80/20 → learning algorithm → train → evaluate → accept |
| แสดง Accuracy Rate ของชุดฝึกสอน | ✅ | ⬜ | [`Experiment Comparison/comparison.md`](Experiment%20Comparison/comparison.md) (train/val acc ทุก Round, `comparison_history.csv` ทุก epoch) |
| แบ่ง Train/Validation 80% : 20% | ✅ | ⬜ | `train_test_split(test_size=0.2, stratify, random_state=42)` → train 50,653 / val 12,664 |
| แนะนำสมาชิกในกลุ่ม | ⬜ | ⬜ | |

## ชุดข้อมูล

ชุดข้อมูลจากอาจารย์ (ไม่อยู่ใน repo นี้) โครงสร้าง `ThaiCharacter Dataset/round2/<รหัสคลาส>/*.jpg`

| รายการ | ค่า |
|---|---|
| จำนวนคลาส | 72 (รหัส 161–249 ไม่ต่อเนื่อง; ไม่มีไฟล์ระบุว่ารหัสใดคืออักษรตัวใด) |
| จำนวนภาพ | 62,707 ภาพ (JPEG เกรย์สเกล 1 ช่อง) |
| ขนาดภาพ | เล็กและไม่คงที่: กว้างเฉลี่ย ~16 px, สูงเฉลี่ย ~21 px (ต่ำสุด ~2x5) |
| ความไม่สมดุล | น้อยสุด 1 ภาพ (คลาส 163, 177), มากสุด 5,025 ภาพ (คลาส 210) |
| หลังเติม augmentation | 63,317 ภาพ (เติม 610 ภาพให้ทุกคลาสมี ≥ 50) → train 50,653 / val 12,664 |

ชุดทดสอบเพิ่มเติม (`datasets for testing/synthetic_test_set`, ไม่อยู่ใน repo): **สร้างขึ้นเองโดยคนในทีม** จากฟอนต์ดิจิทัล 432 ภาพ 72 คลาส
**ไม่ได้ใช้เทรนหรือ validate เลย** (ชุดเทรนคือของอาจารย์ทั้งหมด แบ่ง train 80% / validation 20%) — ภาพของคลาสสระบน/ล่างและวรรณยุกต์ในชุดนี้
มีวงกลมจุดไข่ปลา (◌) ที่ฟอนต์วาดให้อัตโนมัติ ซึ่งไม่มีในชุดของอาจารย์ (ชุดเทรนเป็นชิ้นเครื่องหมายเดี่ยว ๆ) จึงเป็นข้อจำกัดของชุดที่ทีมสร้าง
ไม่ใช่ตัวแทนของชุดทดสอบจริงโดยตรง · ชุดเทรนน่าจะครอปมาจากแบบฟอร์มลายมือ/สแกน (สันนิษฐานจากชื่อไฟล์ ไม่มีเอกสารยืนยัน)

## เทคนิคที่ใช้ (ทุก Round ตั้งแต่ Round 2)

| เทคนิค | รายละเอียด |
|---|---|
| Transfer Learning | ResNet18 pretrained ImageNet, `conv1` ใหม่รับ 1 ช่อง, `fc` = Dropout(0.3) + Linear(72) |
| Data Augmentation | หมุนสุ่ม ±15°, Gaussian noise (var 0.005) ที่ 50% เติมให้ทุกคลาสมี ≥ 50 ภาพ |
| Standardize | `(x − mean) / std` จาก train set, เก็บใน `norm_stats.json` เพื่อใช้ค่าเดียวกันตอน inference |
| Stratified split | 80/20 คงสัดส่วนคลาส (`random_state=42`) |
| Class weight | ถ่วงน้ำหนัก loss แก้ class imbalance |
| Optimizer / Scheduler | Adam lr=1e-4, weight decay 1e-4, `ReduceLROnPlateau` (factor 0.5, patience 3) |
| Early stopping + best checkpoint | เพดาน 60 epoch, patience 10, เก็บเฉพาะโมเดลที่ val_acc สูงสุด |

## พัฒนาการแต่ละ Round

| Round | ทำอะไร | val_acc | synthetic (432 ภาพ) |
|---|---|---|---|
| [`Round 1`](Round%201) | เวอร์ชันแรก: ResNet18, 32x32, หาร 255 อย่างเดียว, split สุ่ม, 25 epoch, เก็บ epoch สุดท้าย (Colab) | 98.48% | 53.70% |
| [`Round 2`](Round%202) | **ตัวส่งปัจจุบัน** — 64x64, ตัด maxpool, dropout, standardize, stratify, weight decay, scheduler, class weight, early stopping, best checkpoint (Colab) | **98.79%** | 68.75% |
| [`Round 3`](Round%203) | ทดลอง 224x224 + คืน maxpool, lazy dataset, เทรน local ด้วย DirectML (AMD RX 7600S) | 98.54% | 69.91% |
| [`Round 4`](Round%204) | ทดลองขนาดภาพ 64/96/128 และเปิด/ปิด maxpool + ทำซ้ำด้วย seed ต่างกัน (6 รัน) | 98.40–98.53% | 68.29–71.06% |
| [`Round 5`](Round%205) | augmentation ใหม่กับทุกภาพของชุด train (หมุน/ย่อขยาย/เลื่อน, ปรับความหนาเส้น, ลบส่วนของภาพ) เทรนบน Colab (1 รอบ) | 98.53% | 69.91% |

**ข้อสรุปตอนนี้:** ตั้งแต่ Round 2 เป็นต้นไป ทุกแบบให้ผลใกล้กันมาก (synthetic ต่างกันไม่กี่ภาพ ซึ่งเทียบเท่าความแกว่งระหว่างรอบของแบบเดียวกัน
เช่น `img96_nomp` ทำซ้ำ 3 รอบได้ 68.98–71.06%, เฉลี่ย 70.29%) จึงยังสรุปไม่ได้ว่าขนาดภาพ/maxpool แบบไหนดีกว่าจริง
Round 5 (augmentation ใหม่) ให้ val/synthetic ใกล้เคียงกัน แต่ **ทนภาพแหว่ง/เส้นบางได้ดีกว่ามาก** (เช่น เส้นบางลง 1 พิกเซล 82.7% เทียบ 56.4%) · ตัวส่งยังเป็น Round 2 (ยังไม่ตัดสินใจ) · รายละเอียดครบทุกค่าและผล: [`Experiment Comparison/comparison.md`](Experiment%20Comparison/comparison.md)
(`comparison.csv` = ทุก config + ผล, `comparison_history.csv` = train/val ทุก epoch)

## 📊 เปรียบเทียบผลทุก Round

<!-- COMPARISON:START -->
### ทุกรัน ทุก Round (⭐ = ค่าสูงที่สุดในคอลัมน์นั้น จากทุกรัน)

| Round | รัน | ขนาด | maxpool | epochs | val_acc | synthetic (432 ภาพ) | นาที/epoch | หมายเหตุ |
|---|---|---|---|---|---|---|---|---|
| Round 1 | `round1` | 32 | on | 25/25 | 98.48% | 53.70% (232/432) | - | เวอร์ชันแรก (เก็บโมเดล epoch สุดท้าย) |
| Round 2 | `round2` | 64 | off | 41/60 | **98.79%** ⭐ | 68.75% (297/432) | - | ตัวส่งปัจจุบัน |
| Round 3 | `round3_img224` | 224 | on | 46/60 | 98.54% | 69.91% (302/432) | 4.3 | resize 224 + maxpool |
| Round 4 | `img128` | 128 | on | 36/60 | 98.50% | 69.91% (302/432) | 2.0 |  |
| Round 4 | `img64_nomp` | 64 | off | 50/60 | 98.46% | 68.52% (296/432) | 1.8 |  |
| Round 4 | `img96` | 96 | on | 40/60 | 98.40% | 68.29% (295/432) | 1.6 |  |
| Round 4 | `img96_nomp` | 96 | off | 46/60 | 98.53% | 70.83% (306/432) | 2.8 | ต้นฉบับ (ไม่ตั้ง seed) |
| Round 4 | `img96_nomp_s1` | 96 | off | 32/60 | 98.49% | 68.98% (298/432) | 3.2 | ทำซ้ำ seed 1 |
| Round 4 | `img96_nomp_s2` | 96 | off | 44/60 | 98.45% | **71.06% (307/432)** ⭐ | 3.1 | ทำซ้ำ seed 2 |
| Round 5 | `img96_nomp_aug_s1` | 96 | off | 55/60 | 98.53% | 69.91% (302/432) | 2.0 | **augmentation ใหม่** — ทนภาพแหว่งดีกว่า Round 2 และ `img96_nomp` มาก (เทรน 1 รอบ บน Colab) |

### ค่าที่ดีที่สุดของแต่ละ Round (⭐ = สูงที่สุดจากทุก Round)

| Round | val_acc สูงสุด (รัน) | synthetic สูงสุด (รัน) |
|---|---|---|
| Round 1 | 98.48% (`round1`) | 53.70% (`round1`) |
| Round 2 | **98.79%** ⭐ (`round2`) | 68.75% (`round2`) |
| Round 3 | 98.54% (`round3_img224`) | 69.91% (`round3_img224`) |
| Round 4 | 98.53% (`img96_nomp`) | **71.06%** ⭐ (`img96_nomp_s2`) |
| Round 5 | 98.53% (`img96_nomp_aug_s1`) | 69.91% (`img96_nomp_aug_s1`) |

อ่านตารางนี้ด้วยความระมัดระวัง: ความต่างของ val_acc ระหว่างรอบส่วนใหญ่เพียง 0.1–0.4 จุด และ synthetic ของแบบเดียวกันแกว่งระหว่างรอบได้ ~2 จุด (1 ภาพ ≈ 0.23 จุด) จึงยังบอกไม่ได้ว่ารอบไหนดีกว่าจริงจากตัวเลขสองคอลัมน์นี้อย่างเดียว · ชุด val ของ Round 1–2 (เทรนบน Colab) น่าจะไม่ตรงกับ Round 3–5 (Round 1 แบ่งสุ่มไม่ stratify; Round 2 ผมสร้างชุด val ซ้ำแล้วไม่ตรงกับที่ log ไว้) จึงเทียบ val ตรง ๆ ไม่ได้ · ตารางความทนทานต่อภาพแหว่ง/รูปแบบภาพ (ซึ่ง Round 5 ดีขึ้นมาก) อยู่ที่ [`Experiment Comparison/robustness_tests.md`](Experiment%20Comparison/robustness_tests.md) · ทุกค่า config และผลเต็มอยู่ที่ [`Experiment Comparison/comparison.md`](Experiment%20Comparison/comparison.md)
<!-- COMPARISON:END -->

## โครงสร้าง repo

| โฟลเดอร์ / ไฟล์ | เนื้อหา |
|---|---|
| `Round 1/` … `Round 4/` | โค้ด train / inference, `model.pt` และ `README.md` ของแต่ละ Round (อธิบายสิ่งที่ทำและต่างจากรอบก่อน) |
| `Round 4/runs/<ชื่อรัน>/` | ผลของแต่ละการทดลอง: `model.pt`, `norm_stats.json`, `metrics.json`, `synthetic.json` |
| `Experiment Comparison/` | ตารางเปรียบเทียบทุก Round, ประวัติ epoch, log และผล synthetic |
| `GPU_Setup_Guide_AMD.md` | วิธีตั้ง DirectML สำหรับเทรนด้วย GPU AMD บน Windows |

ชุดข้อมูลและชุดทดสอบไม่อยู่ใน repo (ต้องมีเองตามโครงสร้างด้านบนถ้าจะเทรนใหม่)

## วิธีรัน (ตัวส่ง = Round 2)

ติดตั้ง: `pip install -r requirements.txt` (มีคำแนะนำเลือก PyTorch ตามการ์ดจอ NVIDIA / AMD / CPU อยู่ในไฟล์)

Inference (รันจากในโฟลเดอร์ `Round 2`):
```bash
cd "Round 2"
python TestingCNN.py path/to/image.png            # ภาพเดียว -> พิมพ์คลาสและความมั่นใจ
python TestingCNN.py path/to/test_folder          # ทั้งโฟลเดอร์ -> out.csv
```
ถ้าโฟลเดอร์ทดสอบแยกตามชื่อคลาส (เช่น `.../161/xxx.png`) จะพิมพ์ Accuracy ให้อัตโนมัติ

Train: Round 1–2 แนะนำรันบน Google Colab (GPU) · Round 3–4 รันบนเครื่องด้วย DirectML — ดู `README.md` ในโฟลเดอร์ของแต่ละ Round และ `GPU_Setup_Guide_AMD.md`
