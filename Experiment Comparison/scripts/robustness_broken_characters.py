"""ทดสอบความทนทานต่อ "การทำให้แหว่ง" (ลบพิกเซล/ก้อน/ขอบ, เส้นบางลง, รอยขาด) — ผลอยู่ใน robustness_tests.md

    cd "Round 4"
    python "../Experiment Comparison/scripts/robustness_broken_characters.py" "../Round 2"
    python "../Experiment Comparison/scripts/robustness_broken_characters.py" runs/img96_nomp

ต้องรันจากในโฟลเดอร์ Round 4 (ใช้ evaluate_synthetic.py ของโฟลเดอร์นั้นโหลดโมเดล) ด้วย python ที่มี torch
"""
import sys, os, glob, io, numpy as np, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from PIL import Image, ImageFilter
target = sys.argv[1]
sys.path.insert(0, os.getcwd())
import evaluate_synthetic as ES
T, _ = ES.load_testing_module(target)
root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(ES.__file__)), "..", "datasets for testing", "synthetic_test_set"))
excl = {209,212,213,214,215,216,217,229,231,232,233,234,236,163,177}
files = [(int(os.path.basename(d)), f) for d in sorted(glob.glob(root + "/*")) for f in sorted(glob.glob(d + "/*.png")) if int(os.path.basename(d)) not in excl]
rng = np.random.default_rng(0)
def degrade(g, kind):
    g = g.copy(); h, w = g.shape; ink = g < 128
    if kind.startswith("ลบพิกเซลหมึกสุ่ม"):
        p = float(kind.split()[-1].rstrip('%')) / 100; g[ink & (rng.random(g.shape) < p)] = 255
    elif kind.startswith("ลบสี่เหลี่ยมสุ่ม"):
        frac = float(kind.split()[-1].rstrip('%')) / 100
        for _ in range(2):
            bh, bw = max(1, int(h * frac)), max(1, int(w * frac)); y = rng.integers(0, h - bh + 1); x = rng.integers(0, w - bw + 1); g[y:y+bh, x:x+bw] = 255
    elif kind.startswith("ลบขอบขวา"):
        frac = float(kind.split()[-1].rstrip('%')) / 100; g[:, int(w * (1 - frac)):] = 255
    elif kind.startswith("ลบขอบบน"):
        frac = float(kind.split()[-1].rstrip('%')) / 100; g[:int(h * frac), :] = 255
    elif kind == "เส้นบางลง (erode 1px)":
        g = np.array(Image.fromarray(g).filter(ImageFilter.MaxFilter(3)))
    elif kind == "รอยขาดแนวนอน 1 เส้น":
        y = rng.integers(h // 3, 2 * h // 3); g[y:y+1, :] = 255
    return g
kinds = ["ไม่ทำลาย (ต้นฉบับ)", "ลบพิกเซลหมึกสุ่ม 10%", "ลบพิกเซลหมึกสุ่ม 20%", "ลบพิกเซลหมึกสุ่ม 30%",
         "ลบสี่เหลี่ยมสุ่ม 15%", "ลบสี่เหลี่ยมสุ่ม 30%", "ลบขอบขวา 15%", "ลบขอบขวา 30%", "ลบขอบบน 15%", "เส้นบางลง (erode 1px)", "รอยขาดแนวนอน 1 เส้น"]
print(f"== {target} | {len(files)} ภาพ (57 คลาส) — accuracy เมื่อภาพถูกทำให้แหว่ง")
tmp = tempfile.mkdtemp(); p = os.path.join(tmp, "x.png")
for k in kinds:
    rng = np.random.default_rng(0); ok = 0
    for c, f in files:
        g = np.array(Image.open(f).convert("L"))
        g = g if k.startswith("ไม่ทำลาย") else degrade(g, k)
        Image.fromarray(g).save(p); pred, _ = T.predict(p); ok += (pred == str(c))
    print(f"  {k:28s} {ok}/{len(files)} = {100*ok/len(files):5.1f}%")
