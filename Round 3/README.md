# Round 3 — ทดลอง IMG_SIZE=224 (เทรนบนเครื่อง local ผ่าน DirectML)

> สถานะ: **เทรนจบแล้ว** (เป็นผลทดลอง ไม่ใช่ตัวส่ง) · รอบก่อน: [`../Round 2/`](<../Round 2>) · รอบถัดไป: [`../Round 4/`](<../Round 4>)
> README เดิมของโฟลเดอร์นี้ (เขียนตอนยังเทรนไม่จบ — เนื้อหาบางส่วนล้าสมัย เช่น ยังบอกว่าไม่มี `model.pt`) อยู่ที่ [`README_original.md`](README_original.md)

## ทำอะไรไปบ้าง

ทดลองว่าถ้า resize ภาพให้เท่ากับ resolution ที่ ResNet18 pretrained คุ้นเคย (224x224) จะได้ผลดีกว่า 64x64 ของ Round 2 ไหม
**ข้อควรระวัง:** ภาพต้นทางจริงมีแค่ ~18x12 พิกเซล การ resize เป็น 224 คือ upscale/interpolate ภาพเบลอให้ใหญ่ขึ้น ไม่ได้เพิ่มรายละเอียดจริง

ลำดับเหตุการณ์:
1. ลองเทรนบน Colab ก่อน — โหลดทุกภาพเป็น float32 array เดียวที่ 224x224 กิน RAM ~12 GB+ → **Colab OOM ล่ม**
   แก้เป็นโหลดแบบ lazy (custom `Dataset`) แล้วเทรนต่อ แต่โควตา GPU ฟรีหมดที่ epoch 46/60
2. ย้ายมาเทรนบนเครื่อง local (GPU AMD Radeon RX 7600S ผ่าน **DirectML**) — ลอง ROCm-on-Windows แล้วแต่ crash
   ตอนใช้ BatchNorm (บั๊กของ wheel) จึงใช้ DirectML ตั้ง venv ที่ `D:\dml-venv-thai` (Python 3.12, `torch-directml`)
3. แก้ให้โหลดภาพเร็วขึ้น: cache ภาพต้นฉบับขนาดเล็กใน RAM + DataLoader 4 worker (ต้องย้ายโค้ดเข้า `main()` + `if __name__ == "__main__":`)
   — **`TrainingCNN.py` ในโฟลเดอร์นี้คือเวอร์ชันที่ใช้เทรนรอบสุดท้ายนี้จริง**

## ต่างจาก Round 2 ยังไง

| จุด | Round 2 | Round 3 |
|---|---|---|
| ขนาดภาพ | 64x64 | **224x224** |
| `maxpool` | ตัดออก | **ใช้ปกติ** (224 ไม่ยุบเกินไป; ได้ feature map 7x7 ก่อน avgpool) |
| การโหลดข้อมูล | พรีโหลดทั้งหมดเป็น array เดียว | **lazy `Dataset`** + cache ภาพต้นฉบับ (RAM น้อยมาก) |
| Augmentation (ภาพที่เติมให้คลาสน้อย) | สร้างครั้งเดียวตอนโหลด ใช้ภาพชุดเดิมทุก epoch | **สุ่มใหม่ทุกครั้งที่ดึงภาพ** (ทุก epoch ได้ภาพต่างกัน) ทำหลัง resize |
| mean/std | คำนวณจาก train ทั้งชุด (0.5592 / 0.4385) | ประเมินจาก **subset สุ่ม 3,000 ภาพ** ของ train (0.5569 / 0.4393) |
| อุปกรณ์เทรน | Colab GPU | **DirectML (RX 7600S)** — เลือก device อัตโนมัติ: DirectML → CUDA → CPU |
| DataLoader | `num_workers=0` (ค่าเริ่มต้น) | 4 worker + `persistent_workers` (ปรับด้วย env `NUM_WORKERS`) |
| ตัวเลือกเสริม | — | env `EPOCHS`, `OPTIMIZER=adam\|sgd` (เพราะ Adam มี op `lerp` ที่ DirectML ไม่รองรับ ต้อง fallback CPU) |

ส่วนอื่นเหมือน Round 2: standardize mean/std, stratify split (`random_state=42`), Adam lr=1e-4 + weight_decay=1e-4,
ReduceLROnPlateau, class weight, early stopping (patience 10, เพดาน 60), เก็บ best checkpoint, seed ไม่ได้ตั้ง

## ผลลัพธ์

| ตัวชี้วัด | ค่า |
|---|---|
| epoch ที่เทรน | 46/60 (early stopping ตัด) · best epoch = 36 |
| ที่ best epoch: train_loss / train_acc | 0.0010 / 99.93% |
| ที่ best epoch: val_loss / **val_acc** | 0.0860 / **98.54%** |
| epoch สุดท้าย (46): train_acc / val_acc | 99.96% / 98.45% |
| เวลาต่อ epoch | ~4.3 นาที (255.7 วินาที) → รวมประมาณ 3 ชั่วโมง 14 นาที |
| synthetic test (432 ภาพ) | **69.91%** (302/432) — วัดจาก `model.pt` ในโฟลเดอร์นี้ (Round 2: 68.75%) |

**สรุป:** ดีกว่า Round 2 บน synthetic แค่ 5 ภาพ (+1.16 จุด) และ val_acc ต่ำกว่าเล็กน้อย (98.54% vs 98.79%) แต่ใช้เวลาเทรนนานกว่ามาก
ความต่างอยู่ในช่วง noise ของชุดทดสอบ 432 ภาพ จึงยังสรุปไม่ได้ว่า 224 ดีกว่าจริง (Round 4 ทดลองต่อ)

ประวัติทุก epoch: [`../Experiment Comparison/logs/round3_img224_terminal_log.txt`](<../Experiment Comparison/logs/round3_img224_terminal_log.txt>)
ตารางเทียบทุกรอบ: [`../Experiment Comparison/comparison.md`](<../Experiment Comparison/comparison.md>)

## ไฟล์ในโฟลเดอร์

| ไฟล์ | หน้าที่ |
|---|---|
| `Net.py` | ResNet18 + `conv1` 1 ช่อง + `maxpool` ปกติ + dropout + `fc` 72 คลาส |
| `TrainingCNN.py` | โค้ด train (lazy Dataset + cache ภาพ, DirectML, early stopping) |
| `TestingCNN.py` | inference — อ่านขนาดภาพ/mean/std จาก `norm_stats.json`; แก้เพิ่ม 2026-09-19: รับ `.jpg/.jpeg/.png/.bmp`, recursive, พิมพ์ accuracy จากชื่อโฟลเดอร์คลาส, stdout UTF-8 |
| `model.pt` | น้ำหนัก best checkpoint (epoch 36) |
| `norm_stats.json` | `{"mean": 0.5569, "std": 0.4393, "img_size": 224}` |
| `out.csv` | ผล inference ที่เคยรันทิ้งไว้ (ไม่ใช่ผลหลัก) |
| `README_original.md` | README เดิม |

## วิธีรัน

ใช้ python จาก venv ที่มี `torch-directml` (รันจากในโฟลเดอร์นี้):
```powershell
D:\dml-venv-thai\Scripts\python.exe TrainingCNN.py
D:\dml-venv-thai\Scripts\python.exe TestingCNN.py "..\datasets for testing\synthetic_test_set"
```
