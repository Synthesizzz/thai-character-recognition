# Round 4 — ทดลองหลายขนาดภาพ / maxpool / seed (ทำซ้ำได้ ผลแยกต่อการทดลอง)

> สถานะ: **ทดลองเสร็จแล้ว** (เป็นผลทดลอง ไม่ใช่ตัวส่ง) · รอบก่อน: [`../Round 3/`](<../Round 3>)
> ตารางเทียบผลทุกรอบ (Round 1–4): [`../Experiment Comparison/comparison.md`](<../Experiment Comparison/comparison.md>)

## ทำอะไรไปบ้าง

ใช้ pipeline ของ Round 3 (lazy Dataset, cache ภาพ, DirectML) เทรนซ้ำหลายแบบโดยปรับแค่ **ขนาดภาพ** และ **maxpool**
แล้ววัดกับ `datasets for testing/synthetic_test_set` (432 ภาพ) ทุกแบบ จากนั้นทำซ้ำแบบที่ดีที่สุดอีก 2 รอบด้วย seed ต่างกัน
เพื่อวัดว่าผลแกว่งระหว่างรอบเท่าไร ผลแต่ละการทดลองเก็บแยกที่ `runs/<ชื่อ>/` ไม่เขียนทับกัน

## ต่างจาก Round 3 ยังไง

| จุด | Round 3 | Round 4 |
|---|---|---|
| ขนาดภาพ | ตายตัว 224 | ตั้งได้ด้วย env **`IMG_SIZE`** |
| `maxpool` | ตายตัว (ใช้) | `Net(use_maxpool=...)` ตั้งด้วย env **`NO_MAXPOOL=1`** — ค่านี้ถูกเก็บใน `norm_stats.json` เพื่อให้ inference สร้างโมเดลตรงกัน |
| Seed | ไม่ตั้ง | env **`SEED`** ควบคุม weight เริ่มต้นของ `conv1`/`fc`, ลำดับ shuffle, augmentation (`seed_worker` สำหรับ worker ของ DataLoader) — การแบ่ง train/val ยังคง `random_state=42` |
| ที่เก็บผล | `model.pt` ข้างสคริปต์ (เขียนทับทุกรอบ) | **`runs/<RUN_NAME>/`**: `model.pt`, `norm_stats.json`, `metrics.json` (train/val ทุก epoch, best epoch, เวลา, seed), `synthetic.json` |
| `TestingCNN.py` | โหลดข้างสคริปต์ | รับ `--run runs/<ชื่อ>` เลือกผลที่จะทดสอบ |
| stdout | — | บังคับ UTF-8 (กัน `UnicodeEncodeError` เมื่อ redirect) |

เครื่องมือที่เพิ่ม:
- `run_experiments.py` — เทรน → วัด synthetic → อัปเดตตาราง ต่อเนื่องหลายแบบ (ข้ามแบบที่เทรนจบแล้ว)
- `evaluate_synthetic.py` — วัด synthetic ของโมเดลใด ๆ จากโฟลเดอร์ของมันเอง (ไม่ต้องก็อป `model.pt`)
- `../Experiment Comparison/compare.py` (ย้ายมาไว้ที่โฟลเดอร์เปรียบเทียบ) — รวมผลทุกรอบเป็น `../Experiment Comparison/comparison.md`, `comparison.csv`, `comparison_history.csv`

## การทดลองและผลลัพธ์

ทุกแบบ: Adam lr=1e-4, early stopping patience 10 (เพดาน 60 epoch), val_acc = ค่าสูงสุดที่ best epoch

| run | size | maxpool | seed | epochs | best ep | best val_acc | นาที/epoch | synthetic (432) |
|---|---|---|---|---|---|---|---|---|
| `img64_nomp` | 64 | off | ไม่ตั้ง | 50 | 40 | 98.46% | 1.7 | 68.52% (296) |
| `img96` | 96 | on | ไม่ตั้ง | 40 | 30 | 98.40% | 1.6 | 68.29% (295) |
| `img128` | 128 | on | ไม่ตั้ง | 36 | 26 | 98.50% | 2.0 | 69.91% (302) |
| `img96_nomp` | 96 | off | ไม่ตั้ง | 46 | 36 | 98.53% | 2.8 | 70.83% (306) |
| `img96_nomp_s1` | 96 | off | 1 | 32 | 22 | 98.49% | 3.2 | 68.98% (298) |
| `img96_nomp_s2` | 96 | off | 2 | 44 | 34 | 98.45% | 3.1 | 71.06% (307) |

`img96_nomp` ทำซ้ำ 3 รอบ: val_acc เฉลี่ย 98.49% (SD 0.04), synthetic เฉลี่ย **70.29%** (SD 1.14 จุด, ช่วง 68.98–71.06%)

**สรุป**
- ทุกแบบให้ val_acc 98.40–98.54% และ synthetic 68.3–71.1% — ความต่างระหว่างขนาดภาพส่วนใหญ่อยู่ในช่วงความแกว่งระหว่างรอบ
  ของแบบเดียวกัน (`img96_nomp` เอง แกว่ง ~2 จุด) จึง **ยังสรุปไม่ได้ว่าขนาดไหนดีกว่าจริง**
- `img96_nomp` มีค่าเฉลี่ย synthetic สูงสุดเล็กน้อย แต่ val_acc ของ Round 2 (98.79%) ยังสูงกว่าทุกแบบในรอบนี้ ข้อมูลสองด้านขัดกัน
- `img128` ให้ผล synthetic เท่า 224 (Round 3) ที่ต้นทุนต่อ epoch ~ครึ่งเดียว
- ตัวส่งยังเป็น `../Round 2/` จนกว่าจะมีหลักฐานที่ชัดกว่านี้

## ไฟล์ในโฟลเดอร์

| ไฟล์/โฟลเดอร์ | หน้าที่ |
|---|---|
| `Net.py` | โมเดล (`use_maxpool` ตั้งได้) |
| `TrainingCNN.py` | โค้ด train (ตัวเลือกผ่าน env: `IMG_SIZE`, `NO_MAXPOOL`, `SEED`, `RUN_NAME`, `OPTIMIZER`, `EPOCHS`, `NUM_WORKERS`) |
| `TestingCNN.py` | inference (`--run runs/<ชื่อ>`) |
| `run_experiments.py`, `evaluate_synthetic.py` (ส่วน `compare.py` ย้ายไป `../Experiment Comparison/`) | เครื่องมือรันการทดลอง วัดผล และเปรียบเทียบ |
| `runs/<ชื่อ>/` | ผลของแต่ละการทดลอง (โมเดล, สถิติ, metrics, synthetic) |
| `experiments_log.txt`, `experiments_log2.txt` | log การเทรนของ `run_experiments.py` (ชุดแรก 4 แบบ / ชุดทำซ้ำ 2 รอบ) |

## วิธีรัน

ใช้ python จาก venv ที่มี `torch-directml` รันจากในโฟลเดอร์นี้:
```powershell
# เทรน 1 แบบ (ตั้งค่าผ่าน env)
$env:IMG_SIZE=96; $env:NO_MAXPOOL=1; $env:RUN_NAME="img96_nomp"; $env:SEED=1
D:\dml-venv-thai\Scripts\python.exe TrainingCNN.py

# หรือรันชุดที่กำหนดไว้ใน run_experiments.py
D:\dml-venv-thai\Scripts\python.exe run_experiments.py img96 img128

# ทดสอบโมเดลของ 1 การทดลอง
D:\dml-venv-thai\Scripts\python.exe TestingCNN.py "..\datasets for testing\synthetic_test_set" --run runs/img96_nomp

# รวมตารางเปรียบเทียบ
D:\dml-venv-thai\Scripts\python.exe "..\Experiment Comparison\compare.py"
```
