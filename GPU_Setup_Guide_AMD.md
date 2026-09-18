# คู่มือ: เปิดใช้ GPU (AMD) กับ PyTorch บนเครื่อง Windows

**สเปคเครื่องตอนทำ**: GPU = AMD Radeon RX 7600S (RDNA3, gfx1102), Windows, Python หลักของเครื่อง = 3.14

---

## สรุปสั้น ๆ (TL;DR)

- การ์ดจอ AMD บน Windows **ใช้ CUDA ไม่ได้** (CUDA เป็นของ NVIDIA เท่านั้น) ต้องใช้ทางอื่น
- ลองมา 2 ทาง: **ROCm-on-Windows** (ทางการของ AMD) กับ **DirectML** (ของ Microsoft, ผ่าน DirectX 12)
- **ROCm-on-Windows: ใช้ไม่ได้จริง** ติดตั้งสำเร็จ detect GPU ได้ แต่ crash ทันทีที่ใช้งานจริง
  (บั๊กที่ยืนยันแล้วว่าอยู่ในตัว wheel ของ AMD เอง ไม่ใช่ปัญหาโค้ดเรา — ดูรายละเอียดท้ายไฟล์)
- **DirectML: ใช้ได้จริง** ทดสอบครบ forward + backward + optimizer + eval + save/load ผ่านหมด
- ต้องใช้ **Python 3.12** (ไม่ใช่ 3.14 ที่เครื่องมีอยู่) เพราะ package ที่เกี่ยวข้องยังไม่รองรับ Python รุ่นใหม่ขนาดนั้น

---

## ขั้นตอนที่ 1: ติดตั้ง Python 3.12 (แยกจาก Python หลักของเครื่อง)

1. โหลดจาก https://www.python.org/downloads/release/python-3120/ (Windows installer 64-bit)
2. ตอนติดตั้งเลือก **"Customize installation"** (ไม่ใช่ "Install Now") เหตุผล: กันชน PATH กับ Python
   หลักที่มีอยู่แล้ว (ในเครื่องนี้คือ 3.14)
   - หน้า Optional Features: ติ๊ก **pip** และ **py launcher** ไว้ (ใช้เรียกผ่าน `py -3.12` ทีหลัง)
   - หน้า Advanced Options: **ไม่ต้องติ๊ก** "Add Python to environment variables"
3. เช็คว่าติดตั้งสำเร็จและเครื่องเห็นทั้งสองเวอร์ชัน:
   ```powershell
   py -0
   ```
   ควรเห็นทั้ง `3.14` (หลัก) และ `3.12` (ที่เพิ่งลง) ในลิสต์

---

## ขั้นตอนที่ 2: สร้าง venv — สำคัญ: ต้องอยู่ที่ path สั้น ๆ

**ห้ามสร้าง venv ไว้ในโฟลเดอร์โปรเจกต์ที่ path ยาว** (เช่นโฟลเดอร์นี้ที่ชื่อยาวมาก) เพราะ package
พวกนี้มีไฟล์ข้างในที่ชื่อยาวมาก (โดยเฉพาะ ROCm) รวมกับ path โปรเจกต์แล้วเกิน 260 ตัวอักษรที่ Windows
จำกัดไว้ (Windows Long Path ไม่ได้เปิดใช้งานโดย default) ทำให้ pip install พังตอน unpack ไฟล์
(เจอปัญหานี้จริงตอนทำ — ดาวน์โหลดไปเกือบ 1.6GB แล้วพังตอนท้าย)

**สร้าง venv ไว้ที่ root ของ drive แทน**:
```powershell
py -3.12 -m venv D:\dml-venv-thai
```
(ตั้งชื่อให้จำง่ายว่าใช้กับโปรเจกต์ไหน ถ้ามีหลายโปรเจกต์ก็แยก venv คนละอันได้)

---

## ขั้นตอนที่ 3: ติดตั้ง PyTorch แบบ DirectML

```powershell
D:\dml-venv-thai\Scripts\python.exe -m pip install --upgrade pip
D:\dml-venv-thai\Scripts\python.exe -m pip install torch-directml
```
คำสั่งนี้จะดึง `torch` เวอร์ชันที่ compatible มาให้อัตโนมัติด้วย (ตอนทำได้ torch 2.4.1)

ติดตั้ง library อื่นที่โปรเจกต์ใช้ (ปรับตามโปรเจกต์จริง):
```powershell
D:\dml-venv-thai\Scripts\python.exe -m pip install scikit-image scikit-learn pandas tqdm
```

---

## ขั้นตอนที่ 4: ทดสอบว่า GPU ใช้งานได้จริง (อย่าข้ามขั้นนี้)

**อย่าเชื่อแค่ device detect ได้** — ต้องทดสอบ forward + backward + optimizer step จริง เพราะ ROCm
ตอนที่ลองก็ detect GPU ได้ปกติ แต่ crash ตอนใช้งานจริง (ดูหัวข้อท้ายไฟล์)

```powershell
D:\dml-venv-thai\Scripts\python.exe -c "
import torch, torch_directml
print('device count:', torch_directml.device_count())
print('device name:', torch_directml.device_name(0))
device = torch_directml.device()

import torch.nn as nn
model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2)).to(device)
x = torch.randn(4, 10, device=device)
y = torch.randint(0, 2, (4,), device=device)
opt = torch.optim.Adam(model.parameters())
loss = torch.nn.functional.cross_entropy(model(x), y)
loss.backward()
opt.step()
print('forward+backward+optimizer: OK')
"
```
ถ้าเห็น `forward+backward+optimizer: OK` แปลว่าใช้งานได้จริง พร้อมเทรนโปรเจกต์จริงได้แล้ว

---

## ขั้นตอนที่ 5: ปรับโค้ดโปรเจกต์ให้เลือก DirectML อัตโนมัติ

แทนที่โค้ด device-selection แบบเดิม:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

ด้วยแบบนี้ (ลอง DirectML ก่อน แล้วค่อย cuda แล้วค่อย cpu เป็น fallback):
```python
try:
    import torch_directml
    if torch_directml.device_count() > 0:
        device = torch_directml.device()
        device_name = torch_directml.device_name(0)
    else:
        raise RuntimeError("ไม่พบ DirectML device")
except (ImportError, RuntimeError):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = str(device)

print(f"ใช้ device: {device_name}")
```

**ข้อควรระวังอื่นตอนพอร์ตโค้ด**:
- ถ้า DataLoader ใช้ `num_workers > 0` และไฟล์เป็น `.py` script ธรรมดา (รันตรง ๆ ไม่ใช่ Jupyter/Colab)
  **ต้องมี `if __name__ == "__main__":` guard** ไม่งั้น multiprocessing จะ error บน Windows —
  ถ้าไม่อยากยุ่งกับการ restructure โค้ด ให้ตั้ง `num_workers=0` ไปเลยง่ายสุด (ช้าลงนิดหน่อยแต่ไม่พัง)

---

## ขั้นตอนที่ 6: วิธีรันจริง

**ผ่าน terminal ตรง ๆ (ชัวร์สุด ไม่ต้องพึ่งการตั้งค่า IDE)**:
```powershell
D:\dml-venv-thai\Scripts\python.exe your_script.py
```

**ผ่าน VSCode** (ถ้าอยากพิมพ์ `python script.py` เฉย ๆ ได้):
1. เปิดโฟลเดอร์โปรเจกต์ใน VSCode
2. `Ctrl+Shift+P` → `Python: Select Interpreter` → `Enter interpreter path...` → พิมพ์
   `D:\dml-venv-thai\Scripts\python.exe`
3. **ปิด terminal เดิมใน VSCode แล้วเปิดใหม่** (สำคัญ — terminal เก่าจะยังผูกกับ interpreter เดิม)
4. เช็คว่าถูก interpreter จริงก่อนรันของจริงเสมอ:
   ```powershell
   python -c "import sys; print(sys.executable)"
   ```
   ต้องเห็น path ของ venv ไม่ใช่ Python หลักของเครื่อง — **ถ้าเห็น device เป็น `cpu` ทั้งที่ทดสอบผ่านมาแล้ว
   ให้เช็คจุดนี้ก่อนเป็นอันดับแรก** เป็นสาเหตุอันดับ 1 ที่เจอจริง (รันด้วย interpreter ผิดตัว
   `import torch_directml` fail แบบเงียบ ๆ แล้ว fallback ไป cpu)

---

## Troubleshooting — ปัญหาที่เจอจริงระหว่างทำ

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `ใช้ device: cpu` ทั้งที่ทดสอบ DirectML ผ่านมาแล้ว | รันด้วย Python/interpreter ผิดตัว (ไม่ใช่ venv ที่ลง `torch-directml` ไว้) | เช็ค `sys.executable` ตามขั้นตอนที่ 6 แล้วเรียกผ่าน path เต็มของ venv |
| `ModuleNotFoundError: No module named 'sklearn'` (หรือ package อื่น) | ลง library ไว้คนละ venv กับที่ใช้รันจริง | เช็คว่า pip install ไปที่ venv ไหน ด้วย `D:\dml-venv-thai\Scripts\python.exe -m pip list` |
| `pip install` ล้มเหลวตอนท้าย พร้อม error พูดถึง path ยาว/`OSError: [Errno 2]` | Windows Long Path + venv อยู่ใน path ยาวเกินไป | ย้าย venv ไปสร้างที่ path สั้น ๆ (เช่น `D:\ชื่อสั้น\`) แล้วรัน pip install ใหม่ (cache เดิมจะถูกใช้ซ้ำ ไม่ต้องโหลดใหม่ทั้งหมด) |
| DataLoader error เกี่ยวกับ multiprocessing/spawn บน Windows | `num_workers > 0` ในสคริปต์ที่ไม่มี `if __name__ == "__main__":` guard | ตั้ง `num_workers=0` หรือ restructure โค้ดให้มี guard |

---

## บันทึกไว้: ทำไมไม่ใช้ ROCm-on-Windows (ทางการของ AMD)

ROCm-on-Windows รองรับ RX 7600 series (gfx1102) อย่างเป็นทางการแล้วในช่วงที่ทำ (ROCm 7.12+) ติดตั้งได้
ด้วยคำสั่ง:
```powershell
python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1102]==2.12.0+rocm7.14.0" "torchvision[device-gfx1102]==0.27.0+rocm7.14.0" "torchaudio==2.11.0+rocm7.14.0"
```
ข้อดีคือใช้ `torch.cuda` API เดิมได้เลย ไม่ต้องแก้โค้ด แต่ **ลองจริงแล้วพบว่า crash** ทันทีที่โมเดลมี
`BatchNorm` (ResNet ทุกรุ่นมี) ข้อความ error คือ:
```
MIOpen(HIP): Error [Compile] ... fatal error: 'type_traits' file not found
```
เช็คแล้วเป็นบั๊กที่รู้จักและยังไม่ถูกแก้ในตัว wheel ของ ROCm-on-Windows เอง (wheel ไม่มี C++ standard
library header ให้ตัว JIT compiler ใช้ตอน compile kernel) มีคนอื่นเจอปัญหาเดียวกันรายงานไว้ใน GitHub
ของ ROCm แล้ว — **ถ้า AMD แก้บั๊กนี้ในอนาคต ROCm จะเป็นตัวเลือกที่ดีกว่า DirectML** (เร็วกว่า, maintain
active กว่า) ลองเช็คใหม่ได้เป็นระยะ ๆ ว่าเวอร์ชันใหม่แก้หรือยัง

โฟลเดอร์ venv ของ ROCm ที่ลองไว้ (ถ้ายังไม่ลบ): `D:\rocm-venv-thai`
