import json
import os
import sys
import csv

import torch
from skimage.io import imread
from skimage.transform import resize

from Net import Net

# ลำดับคลาสต้องตรงกับตอนเทรนเป๊ะ (sorted ชื่อโฟลเดอร์ 161-249 ทั้ง 72 คลาส)
CLASSES = ['161', '162', '163', '164', '167', '168', '169', '170', '171', '173',
           '175', '176', '177', '178', '179', '180', '181', '182', '183', '184',
           '185', '186', '187', '188', '189', '190', '191', '192', '193', '194',
           '195', '196', '197', '199', '200', '201', '202', '203', '204', '205',
           '206', '207', '209', '210', '212', '213', '214', '215', '216', '217',
           '224', '225', '226', '227', '228', '229', '230', '231', '232', '233',
           '234', '236', '240', '241', '242', '243', '244', '245', '246', '247',
           '248', '249']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pt")
NORM_STATS_PATH = os.path.join(SCRIPT_DIR, "norm_stats.json")

if not os.path.isfile(NORM_STATS_PATH):
    raise FileNotFoundError(
        f"ไม่พบ {NORM_STATS_PATH} — ต้องรัน TrainingCNN.py ให้จบก่อน (ไฟล์นี้เก็บ mean/std/img_size "
        "ที่ใช้ตอนเทรน จำเป็นต้องใช้ค่าเดียวกันตอน inference ไม่งั้นโมเดลทำนายผิดเพี้ยน)"
    )
with open(NORM_STATS_PATH, encoding='utf-8') as f:
    _norm_stats = json.load(f)
DATA_MEAN = _norm_stats['mean']
DATA_STD = _norm_stats['std']
IMG_SIZE = _norm_stats['img_size']

# ไม่บังคับ cuda เหมือนตอนเทรน เพราะเครื่อง local (VSCode) มักไม่มี GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net(num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # map_location กันพังตอนโหลดโมเดลที่เทรนจาก GPU มารันบน CPU
model.to(device)
model.eval()


def preprocess(image_path):
    img = imread(image_path, as_gray=True)
    # skimage as_gray=True: ถ้าภาพต้นทางเป็น RGB จะแปลงผ่าน rgb2gray -> ได้ float ช่วง 0-1 อยู่แล้ว
    # แต่ถ้าภาพต้นทางเป็น grayscale เดี่ยวอยู่แล้ว (เหมือนชุดเทรน) จะคืน uint8 ช่วง 0-255 แบบเดิม
    # เช็ค max ก่อนหาร กันหาร 255 ซ้ำสอง (ถ้าหารซ้ำ ค่าจะพังเหลือใกล้ 0 ทุกพิกเซล ทำนายมั่วหมด)
    img = resize(img, (IMG_SIZE, IMG_SIZE), preserve_range=True)  # ต้องเหมือนตอนเทรนเป๊ะ
    if img.max() > 1.0:
        img = img / 255.0
    img = (img - DATA_MEAN) / DATA_STD  # normalize ด้วยค่าเดียวกับตอนเทรน (จาก norm_stats.json)
    img = img.astype('float32')
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor.to(device)


def predict(image_path):
    with torch.no_grad():
        tensor = preprocess(image_path)
        output = model(tensor).squeeze(0)
        prob = torch.softmax(output, dim=0)
        confidence, predicted_idx = torch.max(prob, dim=0)
    return CLASSES[predicted_idx.item()], confidence.item() * 100


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def find_images(folder_path):
    """หาภาพทุกไฟล์ในโฟลเดอร์ (รวมโฟลเดอร์ย่อย) เรียงตามชื่อ"""
    paths = []
    for dirpath, _, filenames in os.walk(folder_path):
        for fname in filenames:
            if fname.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(dirpath, fname))
    return sorted(paths)


def predict_folder(folder_path, output_csv="out.csv"):
    results = []
    n_labeled, n_correct, n_failed = 0, 0, 0
    for path in find_images(folder_path):
        rel = os.path.relpath(path, folder_path)
        try:
            pred_class, conf = predict(path)
        except Exception as e:  # ภาพเสีย/อ่านไม่ได้ ไม่ให้ทั้งรอบล้ม
            print(f"{rel} -> ข้าม (อ่านไม่ได้: {e})")
            n_failed += 1
            continue
        results.append((rel, pred_class, conf))
        print(f"{rel} -> {pred_class} ({conf:.2f}%)")

        # ถ้าภาพอยู่ในโฟลเดอร์ที่ตั้งชื่อตามคลาส (เช่น .../161/xxx.png) ถือเป็นเฉลย ใช้คำนวณ accuracy
        true_class = os.path.basename(os.path.dirname(path))
        if true_class in CLASSES:
            n_labeled += 1
            n_correct += (pred_class == true_class)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_class", "confidence(%)"])
        writer.writerows(results)
    print(f"\nทำนาย {len(results)} ภาพ บันทึกผลลัพธ์ทั้งหมดไว้ที่ {output_csv}")
    if n_failed:
        print(f"ข้ามไป {n_failed} ไฟล์ที่อ่านไม่ได้")
    if n_labeled:
        print(f"Accuracy (จากชื่อโฟลเดอร์ย่อยเป็นเฉลย): {n_correct}/{n_labeled} = {n_correct / n_labeled * 100:.2f}%")


if __name__ == "__main__":
    # terminal บางเครื่อง (เช่น cp1252) พิมพ์ภาษาไทยไม่ได้ -> บังคับ UTF-8 กันแครชหลังทำนายเสร็จ
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    if len(sys.argv) < 2:
        print("วิธีใช้: python TestingCNN.py <path_ไฟล์ภาพ_หรือ_โฟลเดอร์>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        predict_folder(target)
    else:
        pred_class, conf = predict(target)
        print(f"Predicted: {pred_class} ({conf:.2f}%)")
