"""ทดสอบความทนทานต่อ "รูปแบบภาพ" (RGB/RGBA/Binary/กลับสี/สี/JPEG) — ผลอยู่ใน robustness_tests.md

    cd "Round 4"
    python "../Experiment Comparison/scripts/robustness_image_formats.py" "../Round 2"
    python "../Experiment Comparison/scripts/robustness_image_formats.py" runs/img96_nomp

ต้องรันจากในโฟลเดอร์ Round 4 (ใช้ evaluate_synthetic.py ของโฟลเดอร์นั้นโหลดโมเดล) ด้วย python ที่มี torch
"""
import sys, os, glob, io, numpy as np, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from PIL import Image
target = sys.argv[1]
sys.path.insert(0, os.getcwd())
import evaluate_synthetic as ES
T, _ = ES.load_testing_module(target)
root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(ES.__file__)), "..", "datasets for testing", "synthetic_test_set"))
excl = {209,212,213,214,215,216,217,229,231,232,233,234,236,163,177}
files = [(int(os.path.basename(d)), f) for d in sorted(glob.glob(root + "/*")) for f in sorted(glob.glob(d + "/*.png")) if int(os.path.basename(d)) not in excl]
def lerp(g, fg, bg):   # g: 0(ดำ)..255(ขาว) -> สีตัวอักษร..สีพื้น
    a = g[..., None] / 255.0; return (np.array(fg) * (1 - a) + np.array(bg) * a).astype(np.uint8)
def make(im, kind, path):
    g = np.array(im.convert("L"))
    if kind == "gray_original": im.convert("L").save(path)
    elif kind == "rgb (3 ช่อง)": im.convert("RGB").save(path)
    elif kind == "rgba (มี alpha)": im.convert("RGBA").save(path)
    elif kind == "binary 0/255 (threshold 128)": Image.fromarray(((g > 128) * 255).astype(np.uint8)).save(path)
    elif kind == "binary mode '1'": Image.fromarray(g > 128).convert("1").save(path)
    elif kind == "กลับสี (ขาวบนดำ)": Image.fromarray(255 - g).save(path)
    elif kind == "สี: น้ำเงินเข้ม บนเหลืองอ่อน": Image.fromarray(lerp(g, (0,0,160), (255,255,150)), "RGB").save(path)
    elif kind == "สี: แดง บนเขียว (คอนทราสต์ต่ำ)": Image.fromarray(lerp(g, (200,0,0), (0,160,0)), "RGB").save(path)
    elif kind == "JPEG คุณภาพ 30": im.convert("L").save(path, quality=30)
kinds = ["gray_original","rgb (3 ช่อง)","rgba (มี alpha)","binary 0/255 (threshold 128)","binary mode '1'","กลับสี (ขาวบนดำ)","สี: น้ำเงินเข้ม บนเหลืองอ่อน","สี: แดง บนเขียว (คอนทราสต์ต่ำ)","JPEG คุณภาพ 30"]
print(f"== {target} | {len(files)} ภาพ (57 คลาส, ไม่รวมคลาสที่มี ◌) — accuracy ต่อรูปแบบภาพ")
tmp = tempfile.mkdtemp()
for k in kinds:
    ok = 0; err = 0
    for c, f in files:
        ext = ".jpg" if k.startswith("JPEG") else ".png"; p = os.path.join(tmp, "x" + ext)
        make(Image.open(f), k, p)
        try:
            pred, _ = T.predict(p); ok += (pred == str(c))
        except Exception as e: err += 1
    print(f"  {k:34s} {ok}/{len(files)} = {100*ok/len(files):5.1f}%" + (f"  (error {err} ภาพ)" if err else ""))
