"""รันการทดลองหลายขนาดต่อกัน: เทรน -> วัด synthetic -> อัปเดต compare.md  (ข้าม config ที่เทรนจบแล้ว)

    python run_experiments.py                 # รันทุก config ใน EXPERIMENTS
    python run_experiments.py img96 img128    # เฉพาะบางตัว (ชื่อ = RUN_NAME)
รันด้วย python จาก venv ที่มี torch-directml (ใช้ sys.executable ตัวเดียวกับที่เรียกสคริปต์นี้)
"""
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# (RUN_NAME, IMG_SIZE, NO_MAXPOOL[, SEED]) — เรียงจากถูกไปแพง  (ไม่ใส่ SEED = ไม่ตั้ง seed)
EXPERIMENTS = [
    ("img64_nomp", 64, "1"),    # เทียบกับ round 2 (64 ไม่มี maxpool) ภายใต้ pipeline รอบ 3
    ("img96", 96, "0"),
    ("img128", 128, "0"),
    ("img96_nomp", 96, "1"),
    # ทำซ้ำ img96_nomp ด้วย seed ต่างกัน เพื่อวัดความแกว่งระหว่างรอบ (ตัวต้นฉบับข้างบนไม่ได้ตั้ง seed)
    ("img96_nomp_s1", 96, "1", 1),
    ("img96_nomp_s2", 96, "1", 2),
]


def finished(name):
    p = os.path.join(SCRIPT_DIR, "runs", name, "metrics.json")
    if not os.path.isfile(p):
        return False
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("finished", False)


def run(cmd, env=None):
    return subprocess.run([sys.executable, "-u"] + cmd, cwd=SCRIPT_DIR, env=env).returncode


def main():
    only = set(sys.argv[1:])
    for name, size, nomp, *rest in EXPERIMENTS:
        if only and name not in only:
            continue
        if finished(name):
            print(f"== {name}: เทรนจบแล้ว ข้าม", flush=True)
        else:
            print(f"== {name}: เริ่มเทรน {time.strftime('%H:%M:%S')}", flush=True)
            env = dict(os.environ, IMG_SIZE=str(size), NO_MAXPOOL=nomp, RUN_NAME=name, PYTHONIOENCODING="utf-8")
            if rest:
                env["SEED"] = str(rest[0])
            if run(["TrainingCNN.py"], env) != 0:
                print(f"== {name}: เทรนล้มเหลว ข้ามไปตัวถัดไป", flush=True)
                continue
        run(["evaluate_synthetic.py", os.path.join("runs", name)])
        run(["compare.py"])
    print("== ทั้งหมดเสร็จ", time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
