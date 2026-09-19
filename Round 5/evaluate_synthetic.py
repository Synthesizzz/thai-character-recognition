"""วัด accuracy ของโมเดล 1 ชุดกับ datasets for testing/synthetic_test_set

ใช้ได้ 2 แบบ (ไม่ต้องก็อป model.pt ไปไว้ที่อื่น — โหลดจากโฟลเดอร์ของมันเองตรง ๆ):

  1) ผลเทรนของรอบ 4 (runs/<ชื่อ>/ มีแค่ model.pt + norm_stats.json) -> ใช้ TestingCNN.py/Net.py ของโฟลเดอร์นี้
         python evaluate_synthetic.py runs/img96
         ผลเก็บที่ runs/img96/synthetic.json

  2) โฟลเดอร์ที่มีโค้ดของตัวเองครบ (TestingCNN.py + Net.py + model.pt เช่น Round 2/ หรือ Round 3/)
     -> ใช้ TestingCNN.py/Net.py ของโฟลเดอร์นั้นเอง, ไม่แตะไฟล์ในโฟลเดอร์นั้น
         python evaluate_synthetic.py "../Round 3" --out "../Experiment Comparison/results/round3_img224_synthetic.json"
         (ต้องระบุ --out เพราะจะไม่เขียนไฟล์ผลลงในโฟลเดอร์ต้นทาง)
"""
import collections
import importlib.util
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYN_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "datasets for testing", "synthetic_test_set"))


def load_testing_module(target):
    """คืน (module TestingCNN, run_dir, จำนวน path ที่ใช้โค้ดของโฟลเดอร์ไหน)"""
    target_abs = os.path.normpath(os.path.join(SCRIPT_DIR, target))
    own_code = os.path.join(target_abs, "TestingCNN.py")
    if os.path.isfile(own_code):
        code_dir = target_abs               # โฟลเดอร์มีโค้ดของตัวเอง
        argv_run = []                       # TestingCNN ของมันหา model.pt ข้างตัวเองอยู่แล้ว
    else:
        code_dir = SCRIPT_DIR               # ใช้โค้ดของโฟลเดอร์รอบ 4 + ชี้ --run ไปที่ผลเทรน
        argv_run = ["--run", target]
    sys.argv = [sys.argv[0]] + argv_run
    sys.path.insert(0, code_dir)
    os.chdir(code_dir)  # โค้ดเก่า (เช่น round 1) โหลด model.pt ด้วย path สัมพัทธ์กับโฟลเดอร์ที่รัน
    spec = importlib.util.spec_from_file_location("TestingCNN_" + str(abs(hash(code_dir))), os.path.join(code_dir, "TestingCNN.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, code_dir


def main():
    args = sys.argv[1:]
    target = args[0] if args and not args[0].startswith("--") else "."
    out = None
    if "--out" in args:
        out = os.path.normpath(os.path.join(SCRIPT_DIR, args[args.index("--out") + 1]))
    T, code_dir = load_testing_module(target)
    if out is None:
        if os.path.abspath(code_dir) != SCRIPT_DIR:
            sys.exit("โฟลเดอร์ที่มีโค้ดของตัวเองต้องระบุ --out (จะไม่เขียนไฟล์ผลลงในโฟลเดอร์ต้นทาง)")
        out = os.path.join(T.RUN_DIR, "synthetic.json")

    exts = getattr(T, "IMAGE_EXTS", (".jpg", ".jpeg", ".png", ".bmp"))
    ok = n = 0
    wrong = collections.Counter()
    t0 = time.time()
    for cls in sorted(os.listdir(SYN_ROOT)):
        cdir = os.path.join(SYN_ROOT, cls)
        if not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            if not f.lower().endswith(exts):
                continue
            pred, _ = T.predict(os.path.join(cdir, f))
            n += 1
            ok += (pred == cls)
            if pred != cls:
                wrong[cls] += 1
    result = {
        "model_dir": getattr(T, "RUN_DIR", code_dir), "code_dir": code_dir, "img_size": getattr(T, "IMG_SIZE", None),
        "use_maxpool": getattr(T, "USE_MAXPOOL", True),
        "n": n, "correct": ok, "acc": ok / n if n else None,
        "wrong_by_class": dict(sorted(wrong.items())), "eval_sec": time.time() - t0,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"[{target}] synthetic acc: {ok}/{n} = {ok / n * 100:.2f}%  -> {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
