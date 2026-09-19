"""รวมทุกการทดลอง (Round 1–4) เป็นตารางเปรียบเทียบเดียว -> ../Experiment Comparison/

    python compare.py        # รันจากที่ไหนก็ได้ (ใช้แค่ Python มาตรฐาน ไม่ต้องใช้ torch)

ผลที่ได้
- comparison.csv          1 แถว = 1 การทดลอง: คำอธิบายรอบ, ค่าที่ตั้งทุกอย่าง (config), ผลลัพธ์ (train/val/synthetic), path
- comparison.md           ตารางอ่านง่าย + อธิบายว่าแต่ละ Round ทำอะไรไปบ้าง
- comparison_history.csv  ค่า train/val ทุก epoch ของทุกรัน

แหล่งข้อมูล
- Round 4: runs/*/metrics.json, norm_stats.json, synthetic.json (เก็บอัตโนมัติตอนเทรน/วัดผล)
- Round 2 / Round 1: parse บรรทัด "Epoch N/M ..." จาก ../output_colab_round2.ipynb / ../output_colab.ipynb (log ของ Colab)
- Round 3: Experiment Comparison/logs/round3_img224_terminal_log.txt (คัดจาก terminal) + ค่าจากโค้ดใน ../Round 3
- synthetic ของ Round 1–3: Experiment Comparison/results/*_synthetic.json (วัดจาก model.pt ในโฟลเดอร์ของแต่ละรอบ
  ด้วย evaluate_synthetic.py)
- ค่า config ของ Round 1–3 (ที่ไม่มีไฟล์บันทึกอัตโนมัติ) อ่านจากโค้ดในโฟลเดอร์ของรอบนั้น แล้วกรอกไว้ใน ROUND_INFO ด้านล่าง
"""
import csv
import glob
import json
import os
import re
import statistics
import sys

# ไฟล์นี้อยู่ใน Experiment Comparison/ (ที่เดียวกับผลที่สร้าง); โฟลเดอร์ Round 1-5 อยู่ระดับเดียวกับโฟลเดอร์นี้
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
OUT_DIR = SCRIPT_DIR

EPOCH_RE = re.compile(
    r"Epoch (\d+)/(\d+) - train_loss: ([\d.]+) - train_acc: ([\d.]+)% - val_loss: ([\d.]+) - val_acc: ([\d.]+)%"
)
ROUND3_LOG = os.path.join(OUT_DIR, "logs", "round3_img224_terminal_log.txt")
ROUND3_SEC_PER_EPOCH = 255.71  # tqdm ท้าย log: 45 epoch ใช้ 3:11:46

# ------------------------------------------------------------------------------------------
# สิ่งที่แต่ละ Round ทำ + ค่า config ที่ "ไม่ได้ต่างกันในแต่ละรัน" (อ่านจากโค้ด TrainingCNN.py / Net.py ของรอบนั้น)
# ------------------------------------------------------------------------------------------
COMMON_MODEL = "ResNet18 pretrained ImageNet (IMAGENET1K_V1); conv1 ใหม่รับ 1 ช่อง (สุ่มค่าเริ่มต้น); fc -> 72 คลาส"

ROUND_INFO = {
    "Round 1": {
        "round_summary": "เวอร์ชันแรกที่เทรนได้จริงบน Colab: ResNet18 pretrained ทำ Transfer Learning, resize 32x32, "
                         "หาร 255 อย่างเดียว, แบ่ง train/val แบบสุ่ม, เทรน 25 epoch แล้วเก็บโมเดล epoch สุดท้าย",
        "changes_vs_previous": "- (รอบแรก)",
        "device": "Google Colab GPU (CUDA)", "model": COMMON_MODEL + "; มี maxpool; fc = Linear ตรง ๆ (ไม่มี dropout)",
        "dropout": 0.0, "normalization": "หาร 255 เป็นช่วง 0-1 เท่านั้น (ไม่ standardize)", "mean_std": "-",
        "optimizer": "Adam", "lr": 1e-4, "weight_decay": 0.0, "lr_scheduler": "ไม่มี",
        "loss": "CrossEntropyLoss (ไม่ถ่วงน้ำหนัก)", "batch_size": 64, "epoch_cap": 25, "early_stopping": "ไม่มี",
        "checkpoint": "epoch สุดท้ายเสมอ", "split": "80/20 สุ่มล้วน (ไม่ stratify)", "split_seed": "ไม่ตั้ง",
        "augmentation": "เติมภาพคลาสที่มี <50 ภาพให้ครบ 50 (หมุนสุ่ม -15..+15° + Gaussian noise var=0.005 โอกาส 50%)",
        "augmentation_timing": "สร้างครั้งเดียวก่อนเทรน (ภาพชุดเดิมทุก epoch)", "data_loading": "พรีโหลดทุกภาพเป็น array เดียว",
        "num_workers": 0, "train_val_samples": "50,653 / 12,664 (รวม 63,317 = 62,707 จริง + 610 augment)",
        "training_seed": "ไม่ตั้ง",
    },
    "Round 2": {
        "round_summary": "ปรับ pipeline หลายจุดพร้อมกันเพื่อลด overfit/เพิ่ม generalization: resize 64, ตัด maxpool, dropout, "
                         "standardize, stratify split, weight decay, LR scheduler, class weight, early stopping, เก็บ best checkpoint "
                         "(เทรนบน Colab) + แก้บั๊ก preprocess() ที่หาร 255 ซ้ำกับภาพ RGB ตอน inference",
        "changes_vs_previous": "- resize 32->64\n- ตัด maxpool (nn.Identity)\n- fc: Linear -> Dropout(0.3)+Linear\n"
                               "- normalize: หาร 255 -> หาร 255 แล้ว (x-mean)/std จาก train set (เก็บ norm_stats.json)\n"
                               "- split: สุ่มล้วน -> stratify + random_state=42\n- Adam: เพิ่ม weight_decay=1e-4\n"
                               "- เพิ่ม ReduceLROnPlateau (factor 0.5, patience 3)\n- loss: เพิ่ม class weight\n"
                               "- epoch: fix 25 -> เพดาน 60 + early stopping (patience 10)\n- เก็บโมเดล: epoch สุดท้าย -> best val_acc\n"
                               "- แก้บั๊ก preprocess() (เช็ค max>1 ก่อนหาร 255)",
        "device": "Google Colab GPU (CUDA)", "model": COMMON_MODEL + "; ตัด maxpool; fc = Dropout(0.3)+Linear",
        "dropout": 0.3, "normalization": "หาร 255 แล้ว standardize (x-mean)/std", "mean_std": "0.5592 / 0.4385 (จาก train ทั้งชุด)",
        "optimizer": "Adam", "lr": 1e-4, "weight_decay": 1e-4, "lr_scheduler": "ReduceLROnPlateau(mode=min, factor=0.5, patience=3)",
        "loss": "CrossEntropyLoss + class weight", "batch_size": 64, "epoch_cap": 60,
        "early_stopping": "patience 10 (ดู val_acc)", "checkpoint": "best val_acc",
        "split": "80/20 stratify", "split_seed": 42,
        "augmentation": "เหมือน Round 1 (เติมให้ครบ 50 ภาพ/คลาส; หมุน ±15° + noise var 0.005 โอกาส 50%)",
        "augmentation_timing": "สร้างครั้งเดียวก่อนเทรน (ภาพชุดเดิมทุก epoch)", "data_loading": "พรีโหลดทุกภาพเป็น array เดียว",
        "num_workers": 0, "train_val_samples": "50,653 / 12,664", "training_seed": "ไม่ตั้ง",
    },
    "Round 3": {
        "round_summary": "ทดลอง IMG_SIZE=224 (เท่า resolution ที่ ResNet18 pretrained คุ้นเคย; ภาพจริงเล็กมาก ~18x12 จึงเป็นการ upscale) "
                         "คืน maxpool, เปลี่ยนเป็น lazy Dataset + cache ภาพต้นฉบับ (แก้ Colab OOM), ย้ายมาเทรน local ด้วย DirectML (AMD RX 7600S)",
        "changes_vs_previous": "- resize 64->224\n- คืน maxpool\n- โหลดข้อมูลแบบ lazy Dataset + cache ภาพต้นฉบับใน RAM\n"
                               "- augmentation สุ่มใหม่ทุกครั้งที่ดึงภาพ (ทุก epoch ต่างกัน) หลัง resize\n"
                               "- mean/std ประเมินจาก subset สุ่ม 3,000 ภาพ\n- เทรนบน DirectML (RX 7600S) แทน Colab\n"
                               "- DataLoader 4 worker + persistent_workers\n- เพิ่มตัวเลือก env (EPOCHS, OPTIMIZER, NUM_WORKERS)",
        "device": "DirectML: AMD Radeon RX 7600S (local)", "model": COMMON_MODEL + "; มี maxpool; fc = Dropout(0.3)+Linear",
        "dropout": 0.3, "normalization": "หาร 255 แล้ว standardize (x-mean)/std", "mean_std": "0.5569 / 0.4393 (subset 3,000 ภาพ)",
        "optimizer": "Adam", "lr": 1e-4, "weight_decay": 1e-4, "lr_scheduler": "ReduceLROnPlateau(mode=min, factor=0.5, patience=3)",
        "loss": "CrossEntropyLoss + class weight", "batch_size": 64, "epoch_cap": 60,
        "early_stopping": "patience 10 (ดู val_acc)", "checkpoint": "best val_acc",
        "split": "80/20 stratify", "split_seed": 42,
        "augmentation": "เติมให้ครบ 50 ภาพ/คลาส (หมุน ±15° + noise var 0.005 โอกาส 50%)",
        "augmentation_timing": "สุ่มใหม่ทุกครั้งที่ดึงภาพ (หลัง resize)", "data_loading": "lazy Dataset + cache ภาพต้นฉบับใน RAM",
        "num_workers": 4, "train_val_samples": "50,653 / 12,664", "training_seed": "ไม่ตั้ง",
    },
    "Round 4": {
        "round_summary": "ใช้ pipeline ของ Round 3 ทดลองหลายขนาดภาพ (64/96/128) และเปิด/ปิด maxpool แล้ววัด synthetic ทุกแบบ "
                         "จากนั้นทำซ้ำ img96_nomp ด้วย seed ต่างกัน 2 รอบเพื่อวัดความแกว่งระหว่างรอบ (ผลแยกที่ runs/<ชื่อ>/)",
        "changes_vs_previous": "- IMG_SIZE ตั้งได้ด้วย env\n- maxpool เปิด/ปิดได้ (NO_MAXPOOL=1) และเก็บค่าใน norm_stats.json\n"
                               "- เพิ่มตัวเลือก SEED (weight เริ่มต้น, shuffle, augmentation) — การแบ่ง train/val ยัง random_state=42\n"
                               "- เก็บผลแยกต่อการทดลอง runs/<ชื่อ>/ (model.pt, norm_stats.json, metrics.json, synthetic.json)\n"
                               "- เพิ่ม evaluate_synthetic.py, run_experiments.py, compare.py",
        "device": "DirectML: AMD Radeon RX 7600S (local)", "model": COMMON_MODEL + "; maxpool ตามแต่ละรัน; fc = Dropout(0.3)+Linear",
        "dropout": 0.3, "normalization": "หาร 255 แล้ว standardize (x-mean)/std", "mean_std": None,  # อ่านจาก norm_stats.json ต่อรัน
        "optimizer": "Adam", "lr": 1e-4, "weight_decay": 1e-4, "lr_scheduler": "ReduceLROnPlateau(mode=min, factor=0.5, patience=3)",
        "loss": "CrossEntropyLoss + class weight", "batch_size": 64, "epoch_cap": 60,
        "early_stopping": "patience 10 (ดู val_acc)", "checkpoint": "best val_acc",
        "split": "80/20 stratify", "split_seed": 42,
        "augmentation": "เติมให้ครบ 50 ภาพ/คลาส (หมุน ±15° + noise var 0.005 โอกาส 50%)",
        "augmentation_timing": "สุ่มใหม่ทุกครั้งที่ดึงภาพ (หลัง resize)", "data_loading": "lazy Dataset + cache ภาพต้นฉบับใน RAM",
        "num_workers": 4, "train_val_samples": "50,653 / 12,664", "training_seed": None,  # ต่อรัน
    },
    "Round 5": {
        "round_summary": "เปลี่ยนเฉพาะ Data Augmentation: สุ่มแปลงภาพเทรน *ทุกภาพ* ระหว่างเทรน (หมุน/ย่อขยาย/เลื่อน, ปรับความหนาเส้น, ลบส่วนของภาพ) "
                         "เฉพาะชุด train ใช้ข้อมูลของอาจารย์เท่านั้น เทรนที่ 96x96 ไม่มี maxpool 2 รอบ (seed 1 บน Colab, seed 2 บนเครื่อง DirectML) เทียบกับ img96_nomp ของ Round 4",
        "changes_vs_previous": "- Augmentation ใหม่กับทุกภาพของชุด train (เดิมเฉพาะภาพเติม ~1%): หมุน ±10°/ย่อขยาย 0.9–1.1/เลื่อน ±6% (50%), "
                               "ปรับความหนาเส้น 1–2% ของขนาดภาพ (30%), ลบส่วนของภาพ 1–2 ก้อน ก้อนละ 10–35% (40%)\n"
                               "- ชุด val ไม่ถูก augment; โมเดล/optimizer/loss/scheduler/early stopping คงเดิม\n"
                               "- เพิ่ม env AUGMENT, AUG_PREVIEW; TestingCNN แก้โหลดน้ำหนักจาก CUDA (map_location='cpu') และภาพ 1-bit (bool->uint8)",
        "device": "ต่อรัน (ดูคอลัมน์ device ใน comparison.csv)", "model": COMMON_MODEL + "; ไม่มี maxpool (96x96); fc = Dropout(0.3)+Linear",
        "dropout": 0.3, "normalization": "หาร 255 แล้ว standardize (x-mean)/std", "mean_std": None,  # อ่านจาก norm_stats.json ต่อรัน
        "optimizer": "Adam", "lr": 1e-4, "weight_decay": 1e-4, "lr_scheduler": "ReduceLROnPlateau(mode=min, factor=0.5, patience=3)",
        "loss": "CrossEntropyLoss + class weight", "batch_size": 64, "epoch_cap": 60,
        "early_stopping": "patience 10 (ดู val_acc)", "checkpoint": "best val_acc",
        "split": "80/20 stratify", "split_seed": 42,
        "augmentation": "ภาพเติม (ครบ 50/คลาส): หมุน ±15° + noise; ทุกภาพ train: หมุน ±10°/ย่อขยาย/เลื่อน 50%, ปรับความหนาเส้น 30%, ลบส่วนของภาพ 40%",
        "augmentation_timing": "สุ่มใหม่ทุกครั้งที่ดึงภาพ (เฉพาะชุด train)", "data_loading": "lazy Dataset + cache ภาพต้นฉบับใน RAM",
        "num_workers": 2, "train_val_samples": "50,653 / 12,664", "training_seed": None,  # ต่อรัน
    },
}
ROUND_ORDER = ["Round 1", "Round 2", "Round 3", "Round 4", "Round 5"]

# อุปกรณ์ที่ใช้เทรนของแต่ละรัน Round 5: (device, ป้ายสั้น, num_workers)
R5_RUN_DEVICE = {
    "img96_nomp_aug_s1": ("Google Colab GPU (CUDA)", "Colab GPU", 2),
    "img96_nomp_aug_s2": ("DirectML: AMD Radeon RX 7600S (local)", "เครื่อง local DirectML", 4),
}


def load(path):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def hist_from_lines(lines):
    hist, cap = {}, None
    for line in lines:
        m = EPOCH_RE.search(line)
        if m:
            ep, cap = int(m.group(1)), int(m.group(2))
            hist[ep] = {"train_loss": float(m.group(3)), "train_acc": float(m.group(4)) / 100,
                        "val_loss": float(m.group(5)), "val_acc": float(m.group(6)) / 100}
    return (hist or None), cap


def parse_notebook(name):
    d = load(os.path.join(PROJECT_DIR, name))
    if not d:
        return None, None
    lines = []
    for c in d.get("cells", []):
        for o in c.get("outputs", []):
            t = o.get("text") or []
            t = "".join(t) if isinstance(t, list) else t
            lines += t.replace("\r", "\n").split("\n")
    return hist_from_lines(lines)


def parse_log_text(path):
    if not os.path.isfile(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        return hist_from_lines(f.read().split("\n"))


def at(hist, ep):
    return {k: hist[ep][k] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}


def from_history(hist, best_epoch=None):
    last_ep = max(hist)
    if best_epoch is None:
        best_epoch = max(hist, key=lambda e: hist[e]["val_acc"])
    return {"epochs": last_ep, "best_epoch": best_epoch, "best": at(hist, best_epoch), "last": at(hist, last_ep)}


def syn_of(d):
    if not d:
        return {"correct": None, "n": None, "acc": None, "wrong_classes": None, "worst": None}
    w = d.get("wrong_by_class", {})
    worst = ", ".join(f"{k}:{v}" for k, v in sorted(w.items(), key=lambda kv: (-kv[1], kv[0]))[:5])
    return {"correct": d["correct"], "n": d["n"], "acc": d["acc"], "wrong_classes": len(w), "worst": worst}


def build_rows():
    rows, histories = [], {}

    def add(rnd, run, path, size, maxpool, seed, hist_info, spe, syn, mean_std=None, training_seed=None, note="",
            device=None, num_workers=None):
        info = ROUND_INFO[rnd]
        r = {"round": rnd, "run": run, "path": path, "seed": seed}
        r.update({k: info[k] for k in ("round_summary", "changes_vs_previous", "device", "model", "dropout", "normalization",
                                        "optimizer", "lr", "weight_decay", "lr_scheduler", "loss", "batch_size",
                                        "early_stopping", "checkpoint", "split", "split_seed", "augmentation",
                                        "augmentation_timing", "data_loading", "num_workers", "train_val_samples")})
        r.update({"img_size": size, "maxpool": maxpool, "epoch_cap": info["epoch_cap"],
                  "mean_std": mean_std if mean_std is not None else info["mean_std"],
                  "training_seed": training_seed if training_seed is not None else info["training_seed"],
                  "sec_per_epoch": spe, "syn": syn_of(syn), "note": note})
        r.update(hist_info)
        if device is not None:
            r["device"] = device          # ต่อรัน (เช่น Round 5: seed 1 บน Colab, seed 2 บนเครื่อง DirectML)
        if num_workers is not None:
            r["num_workers"] = num_workers
        rows.append(r)

    h1, cap1 = parse_notebook("output_colab.ipynb")
    if h1:
        histories["round1"] = h1
        add("Round 1", "round1", "Round 1", 32, "on", "-", from_history(h1, best_epoch=max(h1)), None,
            load(os.path.join(OUT_DIR, "results", "round1_synthetic.json")),
            note="เก็บโมเดล epoch สุดท้าย (best=last); ค่า train/val จาก log ของ Colab; synthetic วัดจาก model.pt ในโฟลเดอร์นี้")
    h2, cap2 = parse_notebook("output_colab_round2.ipynb")
    if h2:
        histories["round2"] = h2
        add("Round 2", "round2", "Round 2", 64, "off", "-", from_history(h2), None,
            load(os.path.join(OUT_DIR, "results", "round2_synthetic.json")),
            note="ตัวส่งปัจจุบัน; ค่า train/val จาก log ของ Colab; synthetic วัดจาก model.pt ในโฟลเดอร์นี้")
    h3, cap3 = parse_log_text(ROUND3_LOG)
    if h3:
        histories["round3_img224"] = h3
        add("Round 3", "round3_img224", "Round 3", 224, "on", "-", from_history(h3), ROUND3_SEC_PER_EPOCH,
            load(os.path.join(OUT_DIR, "results", "round3_img224_synthetic.json")),
            note="ประวัติทุก epoch จาก terminal log; เวลา/epoch จากแถบ tqdm")

    for d in sorted(glob.glob(os.path.join(PROJECT_DIR, "Round 4", "runs", "*"))):
        m = load(os.path.join(d, "metrics.json"))
        ns = load(os.path.join(d, "norm_stats.json")) or {}
        if not m:
            continue

        def pick(k, m=m):
            return {"train_loss": m["train_loss"][k], "train_acc": m["train_acc"][k],
                    "val_loss": m["val_loss"][k], "val_acc": m["val_acc"][k]}

        histories[m["run"]] = {e + 1: pick(e) for e in range(len(m["val_acc"]))}
        seed = m.get("seed")
        ms = f"{ns['mean']:.4f} / {ns['std']:.4f} (subset 3,000 ภาพ)" if "mean" in ns else "-"
        add("Round 4", m["run"], f"Round 4/runs/{m['run']}", m["img_size"], "on" if m["use_maxpool"] else "off",
            seed if seed is not None else "ไม่ตั้ง",
            {"epochs": m["epochs_run"], "best_epoch": m["best_epoch"], "best": pick(m["best_epoch"] - 1),
             "last": pick(len(m["val_acc"]) - 1)},
            m.get("sec_per_epoch"), load(os.path.join(d, "synthetic.json")), mean_std=ms,
            training_seed=f"seed={seed}" if seed is not None else "ไม่ตั้ง",
            note="" if m["finished"] else "ยังเทรนไม่จบ / ถูกหยุดกลางคัน")

    # ---- Round 5: ../Round 5/runs/* (ต้องมี metrics.json; โฟลเดอร์ที่มีแค่ภาพตัวอย่าง/norm_stats ข้ามไป)
    for d in sorted(glob.glob(os.path.join(PROJECT_DIR, "Round 5", "runs", "*"))):
        m = load(os.path.join(d, "metrics.json"))
        ns = load(os.path.join(d, "norm_stats.json")) or {}
        if not m:
            continue

        def pick5(k, m=m):
            return {"train_loss": m["train_loss"][k], "train_acc": m["train_acc"][k],
                    "val_loss": m["val_loss"][k], "val_acc": m["val_acc"][k]}

        histories[m["run"]] = {e + 1: pick5(e) for e in range(len(m["val_acc"]))}
        seed = m.get("seed")
        ms = f"{ns['mean']:.4f} / {ns['std']:.4f} (subset 3,000 ภาพ)" if "mean" in ns else "-"
        add("Round 5", m["run"], f"Round 5/runs/{m['run']}", m["img_size"], "on" if m["use_maxpool"] else "off",
            seed if seed is not None else "ไม่ตั้ง",
            {"epochs": m["epochs_run"], "best_epoch": m["best_epoch"], "best": pick5(m["best_epoch"] - 1),
             "last": pick5(len(m["val_acc"]) - 1)},
            m.get("sec_per_epoch"), load(os.path.join(d, "synthetic.json")), mean_std=ms,
            training_seed=f"seed={seed}" if seed is not None else "ไม่ตั้ง",
            note=("" if m["finished"] else "ยังเทรนไม่จบ / ถูกหยุดกลางคัน") + f" [{R5_RUN_DEVICE.get(m['run'], ('', ''))[1]}; augmentation ใหม่]",
            device=R5_RUN_DEVICE.get(m["run"], (None, ""))[0], num_workers=R5_RUN_DEVICE.get(m["run"], (None, "", None))[2] if len(R5_RUN_DEVICE.get(m["run"], ())) > 2 else None)
    return rows, histories


CSV_COLS = [
    ("round", None), ("run", None), ("path", None), ("round_summary", None), ("changes_vs_previous", None),
    # --- config ---
    ("img_size", None), ("maxpool", None), ("model", None), ("dropout", None), ("normalization", None), ("mean_std", None),
    ("optimizer", None), ("lr", None), ("weight_decay", None), ("lr_scheduler", None), ("loss", None), ("batch_size", None),
    ("epoch_cap", None), ("early_stopping", None), ("checkpoint", None), ("split", None), ("split_seed", None),
    ("training_seed", None), ("train_val_samples", None), ("augmentation", None), ("augmentation_timing", None),
    ("data_loading", None), ("num_workers", None), ("device", None),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, histories = build_rows()

    header = [c for c, _ in CSV_COLS] + [
        "epochs_used", "best_epoch", "best_train_loss", "best_train_acc", "best_val_loss", "best_val_acc",
        "last_train_loss", "last_train_acc", "last_val_loss", "last_val_acc", "min_per_epoch", "total_min",
        "syn_correct", "syn_n", "syn_acc", "syn_classes_with_errors", "syn_worst_classes", "full_path", "note"]
    with open(os.path.join(OUT_DIR, "comparison.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            b, l, s, spe = r["best"], r["last"], r["syn"], r["sec_per_epoch"]
            w.writerow([r[c] for c, _ in CSV_COLS] + [
                r["epochs"], r["best_epoch"], b["train_loss"], b["train_acc"], b["val_loss"], b["val_acc"],
                l["train_loss"], l["train_acc"], l["val_loss"], l["val_acc"],
                None if spe is None else round(spe / 60, 2), None if spe is None else round(spe * r["epochs"] / 60, 1),
                s["correct"], s["n"], s["acc"], s["wrong_classes"], s["worst"],
                os.path.join(PROJECT_DIR, r["path"]), r["note"]])

    with open(os.path.join(OUT_DIR, "comparison_history.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["run", "epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for run, h in histories.items():
            for ep in sorted(h):
                w.writerow([run, ep, h[ep]["train_loss"], h[ep]["train_acc"], h[ep]["val_loss"], h[ep]["val_acc"]])

    # ---------- Markdown ----------
    pct = lambda x: "-" if x is None else f"{x * 100:.2f}%"
    f4 = lambda x: "-" if x is None else f"{x:.4f}"
    L = ["# เปรียบเทียบผลการทดลองทั้งหมด Round 1–5 (สร้างโดย `compare.py` — อย่าแก้มือ)", "",
         "ไฟล์ข้อมูลเต็ม: `comparison.csv` (ทุกค่า config + ผลลัพธ์ + path เต็ม) · `comparison_history.csv` (train/val ทุก epoch)", "",
         "## แต่ละ Round ทำอะไรไปบ้าง", ""]
    for rnd in ROUND_ORDER:
        info = ROUND_INFO[rnd]
        L += [f"### {rnd}", "", info["round_summary"], "", "**ต่างจากรอบก่อน:**", ""]
        L += [("- " + x.lstrip("- ")) if x.strip() else x for x in info["changes_vs_previous"].split("\n")]
        L += [""]
    L += ["## ผลลัพธ์ (ณ best epoch = checkpoint ที่เก็บ) + synthetic", "",
          "synthetic = `datasets for testing/synthetic_test_set` 432 ภาพ (1 ภาพ ≈ 0.23 จุด — ความต่างไม่กี่ภาพคือ noise)", "",
          "| round | run | size | maxpool | seed | epochs/cap | best ep | train_loss | train_acc | val_loss | val_acc | นาที/epoch | synthetic | path |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        b, s = r["best"], r["syn"]
        spe = "-" if r["sec_per_epoch"] is None else f"{r['sec_per_epoch'] / 60:.1f}"
        syn = "-" if s["acc"] is None else f"{s['acc'] * 100:.2f}% ({s['correct']}/{s['n']})"
        L.append(f"| {r['round']} | {r['run']} | {r['img_size']} | {r['maxpool']} | {r['seed']} | {r['epochs']}/{r['epoch_cap']} | "
                 f"{r['best_epoch']} | {f4(b['train_loss'])} | {pct(b['train_acc'])} | {f4(b['val_loss'])} | {pct(b['val_acc'])} | "
                 f"{spe} | {syn} | `{r['path']}/` |")
    L += ["", "## ผล ณ epoch สุดท้าย (ก่อน early stopping ตัด)", "",
          "| run | epoch สุดท้าย | train_loss | train_acc | val_loss | val_acc |", "|---|---|---|---|---|---|"]
    for r in rows:
        l = r["last"]
        L.append(f"| {r['run']} | {r['epochs']} | {f4(l['train_loss'])} | {pct(l['train_acc'])} | {f4(l['val_loss'])} | {pct(l['val_acc'])} |")

    rep = [r for r in rows if re.fullmatch(r"img96_nomp(_s\d+)?", r["run"])]
    if len(rep) >= 2:
        va = [r["best"]["val_acc"] * 100 for r in rep]
        sy = [r["syn"]["acc"] * 100 for r in rep if r["syn"]["acc"] is not None]
        sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
        L += ["", f"## img96_nomp ทำซ้ำ {len(rep)} รอบ (ต้นฉบับไม่ตั้ง seed, s1 seed=1, s2 seed=2)", "",
              "| ค่า | เฉลี่ย | SD | ต่ำสุด | สูงสุด |", "|---|---|---|---|---|",
              f"| best val_acc | {statistics.mean(va):.2f}% | {sd(va):.2f} | {min(va):.2f}% | {max(va):.2f}% |",
              f"| synthetic | {statistics.mean(sy):.2f}% | {sd(sy):.2f} | {min(sy):.2f}% | {max(sy):.2f}% |"]

    rep5 = [r for r in rows if re.fullmatch(r"img96_nomp_aug(_s\d+)?", r["run"])]
    if len(rep5) >= 2:
        va5 = [r["best"]["val_acc"] * 100 for r in rep5]
        sy5 = [r["syn"]["acc"] * 100 for r in rep5 if r["syn"]["acc"] is not None]
        sd5 = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
        seeds5 = ", ".join(f"{r['run']} (seed {r['seed']})" for r in rep5)
        L += ["", f"## Round 5 (augmentation ใหม่) ทำซ้ำ {len(rep5)} รอบ: {seeds5}", "",
              "| ค่า | เฉลี่ย | SD | ต่ำสุด | สูงสุด |", "|---|---|---|---|---|",
              f"| best val_acc | {statistics.mean(va5):.2f}% | {sd5(va5):.2f} | {min(va5):.2f}% | {max(va5):.2f}% |",
              f"| synthetic | {statistics.mean(sy5):.2f}% | {sd5(sy5):.2f} | {min(sy5):.2f}% | {max(sy5):.2f}% |"]

    L += ["", "## ค่า config ที่ใช้ (ต่อ Round)", "",
          "| ค่า | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |", "|---|---|---|---|---|---|"]
    keys = [("device", "อุปกรณ์เทรน"), ("model", "โมเดล"), ("dropout", "dropout"), ("normalization", "normalize"),
            ("mean_std", "mean / std"), ("optimizer", "optimizer"), ("lr", "learning rate"), ("weight_decay", "weight decay"),
            ("lr_scheduler", "LR scheduler"), ("loss", "loss"), ("batch_size", "batch size"), ("epoch_cap", "epoch สูงสุด"),
            ("early_stopping", "early stopping"), ("checkpoint", "การเก็บโมเดล"), ("split", "แบ่ง train/val"),
            ("split_seed", "seed ตอนแบ่ง"), ("training_seed", "seed ตอนเทรน"), ("train_val_samples", "จำนวน train / val"),
            ("augmentation", "augmentation"), ("augmentation_timing", "augmentation สร้างเมื่อไร"),
            ("data_loading", "การโหลดข้อมูล"), ("num_workers", "DataLoader workers")]
    for k, label in keys:
        vals = [ROUND_INFO[r][k] if ROUND_INFO[r][k] is not None else "ต่อรัน (ดู comparison.csv)" for r in ROUND_ORDER]
        L.append(f"| {label} | " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in vals) + " |")
    L.append("| ขนาดภาพ / maxpool | 32 / on | 64 / off | 224 / on | 64, 96, 128 / ตามรัน (ดู comparison.csv) | 96 / off |")

    L += ["", "## หมายเหตุ", ""] + [f"- **{r['run']}**: {r['note']}" for r in rows if r["note"]]
    with open(os.path.join(OUT_DIR, "comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
