# Round 3 — ทดลอง IMG_SIZE=224 (WIP, กำลังเทรนบนเครื่อง local ผ่าน DirectML)

โฟลเดอร์นี้ทดลองต่อจาก [`../Round 2/`](<../Round 2>) (round 2, val_acc 98.79%, resize 64x64)
ยังไม่ได้เทรนจนจบ — **ยังไม่มี `model.pt` / `norm_stats.json`** ล่าสุดกำลังรันบนเครื่อง local
(Colab หมด GPU quota ระหว่างทดสอบ เลยย้ายมารันเครื่องเอง — ดูหัวข้อ "รันบนเครื่อง local" ด้านล่าง)

## ทำไมถึงลอง 224

ResNet18 pretrained มาจาก ImageNet ที่ 224x224 — ทดลองดูว่าถ้า resize ภาพให้ตรงกับ resolution นี้พอดี
จะช่วยให้ pretrained feature ใช้งานได้เต็มที่ขึ้นไหม **ข้อควรระวัง**: ภาพต้นทางจริงมีแค่ ~18x12 พิกเซล
เท่านั้น (เช็คแล้วตอน debug บั๊ก RGB ก่อนหน้านี้) เพราะงั้นนี่คือการ **upscale/interpolate ภาพเบลอ ๆ ให้ใหญ่ขึ้น
ไม่ได้เพิ่ม detail ที่ไม่มีอยู่จริง** — มีโอกาสได้ผลดีขึ้นแต่ก็มีโอกาสไม่คุ้มกับ compute ที่เสียไป ต้องลองจริงถึงจะรู้

## ต่างจาก Round 2/ ตรงไหนบ้าง

| จุด | Round 2 (`Round 2/`) | Round 3 (ที่นี่) |
|---|---|---|
| Image resize | 64x64 | 224x224 |
| Net.py maxpool | ตัดออก (`nn.Identity()`) | คืนกลับมาใช้ปกติ (เพราะ 224 ไม่ยุบเกินไปเหมือน 64) |
| จุดอื่นทั้งหมด | — | เหมือนกันทุกอย่าง (normalize, stratify, weight_decay, scheduler, class weight, early stopping) |

## รันบนเครื่อง local ผ่าน DirectML (AMD Radeon RX 7600S)

Colab ฟรี GPU quota หมดระหว่างเทรน (ค้างที่ epoch 46/60) เลยย้ายมารันเครื่อง local แทน เครื่องนี้เป็น
**AMD GPU** ไม่มี CUDA ให้ใช้ ลองมาแล้ว 2 ทาง:

| ทาง | ผล |
|---|---|
| ROCm-on-Windows (ทางการของ AMD, รองรับ gfx1102/RX 7600 series) | ❌ ติดตั้งสำเร็จ detect GPU ได้ แต่ **crash ทันทีที่ใช้จริง** — บั๊กที่ยืนยันแล้วในตัว wheel เอง (ไม่มี C++ header ให้ compiler JIT ใช้ตอน compile BatchNorm kernel) ResNet ใช้ BatchNorm ทุกชั้น ชนแน่นอน 100% เป็นปัญหา upstream ของ AMD ไม่ใช่โค้ดเรา |
| **DirectML** (`torch-directml`, ผ่าน DirectX 12) | ✅ **ใช้งานได้จริง** — เทสแล้วครบ forward + backward + optimizer step + scheduler + eval + save/load state_dict ผ่านหมด |

**Setup ที่ใช้อยู่ (ทำไว้ให้แล้ว ไม่ต้องทำซ้ำ):**
- Python 3.12 (แยกจาก Python 3.14 หลักของเครื่อง)
- venv อยู่ที่ `D:\dml-venv-thai` (ตั้งใจวางไว้ path สั้น ๆ นอกโฟลเดอร์โปรเจกต์ — path ยาวเกินไปทำให้ pip
  ติดปัญหา Windows Long Path ตอนติดตั้ง ROCm ครั้งแรก)
- ติดตั้งแล้ว: `torch-directml` (ดึง torch 2.4.1 มาด้วย), `scikit-image`, `scikit-learn`, `pandas`, `tqdm`
- `TrainingCNN.py`/`TestingCNN.py` ปรับ device-selection ให้ลอง DirectML ก่อนอัตโนมัติ (fallback เป็น
  cuda แล้วค่อย cpu) — ไม่ต้องแก้อะไรเพิ่ม
- `num_workers=0` ใน `DataLoader` (ไม่ใช่ 2 เหมือนใน Colab) เพราะรัน `.py` ตรง ๆ บน Windows ต้องมี
  `if __name__ == "__main__":` guard ถึงจะใช้ multiprocessing worker ได้ปลอดภัย ไฟล์นี้ไม่มี guard
  เลยตั้ง 0 ไว้กันพัง (โหลดข้อมูลบน main thread แทน ช้าลงนิดหน่อยแต่ไม่เสี่ยง error)

**วิธีรัน**: ใช้ python จาก venv นี้เรียกสคริปต์ตรง ๆ (ไม่ต้อง activate ก็ได้):
```powershell
D:\dml-venv-thai\Scripts\python.exe TrainingCNN.py
```
(รันจากในโฟลเดอร์นี้ หรือ cd เข้ามาก่อน เพราะ path dataset อ้างอิงแบบ relative)

**ประมาณเวลา**: จาก smoke test (3 training step, batch 64) ได้ ~1 วินาที/step บน RX 7600S ผ่าน DirectML
→ train set ~50,653 ภาพ / 64 = ~792 step/epoch → **ประมาณ 13-15 นาที/epoch** (ไม่รวม eval loop) เทียบกับ
CPU ล้วนที่ประเมินไว้ก่อนหน้าว่า "หลายชั่วโมงถึงหลายวัน" นี่เร็วกว่ามาก ใช้ได้จริง

## ความเสี่ยงที่ต้องระวัง (จากตอนรันบน Colab, ยังเกี่ยวอยู่)

- ~~**RAM**: เสี่ยง Colab ฟรี OOM~~ **เจอ crash จริงแล้ว แก้แล้ว** — เปลี่ยนจากพรีโหลดทั้ง 63,000+ ภาพ
  เป็น float32 array เดียว (~12GB+) เป็น **lazy loading ผ่าน custom `Dataset`** แทน (โหลด/resize/
  normalize ทีละภาพตอน `DataLoader` เรียกจริง) peak memory เหลือแค่ขนาด 1 batch ไม่ว่า resize จะใหญ่
  แค่ไหน — mean/std ก็เปลี่ยนมาประเมินจาก subset สุ่ม 3000 ภาพแทนการโหลดทั้งชุดด้วยเหตุผลเดียวกัน
  (แก้ไว้ทั้งใน notebook หลักและสคริปต์ในโฟลเดอร์นี้)

## ขั้นตอนถัดไป

1. รัน `TrainingCNN.py` ด้วย venv ข้างบนบนเครื่อง local
2. ได้ `model.pt` + `norm_stats.json` ในโฟลเดอร์นี้อัตโนมัติเมื่อเทรนเสร็จ (ไม่ต้องโหลดจาก Colab แล้ว)
3. รัน `TestingCNN.py` sanity-check + ทดสอบกับ `datasets for testing/synthetic_test_set` เทียบกับ
   round 2 (val_acc 98.79%, synthetic 68.75%)
4. ถ้าดีขึ้นชัดเจนพอคุ้มกับเวลาเทรน/compute ที่เสียไป ค่อยตัดสินใจว่าจะใช้แทน `Submit/` หรือไม่

## วิธีรัน

Train:
```bash
D:\dml-venv-thai\Scripts\python.exe TrainingCNN.py
```

Inference (ต้องมี `model.pt` และ `norm_stats.json` ในโฟลเดอร์นี้ก่อน):
```bash
D:\dml-venv-thai\Scripts\python.exe TestingCNN.py path/to/image.jpg          # ทดสอบภาพเดียว
D:\dml-venv-thai\Scripts\python.exe TestingCNN.py path/to/test_folder        # ทดสอบทั้งโฟลเดอร์ -> ได้ out.csv
```
