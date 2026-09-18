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

MODEL_PATH = "model.pt"
IMG_SIZE = 32

# ไม่บังคับ cuda เหมือนตอนเทรน เพราะเครื่อง local (VSCode) มักไม่มี GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net(num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # map_location กันพังตอนโหลดโมเดลที่เทรนจาก GPU มารันบน CPU
model.to(device)
model.eval()


def preprocess(image_path):
    img = imread(image_path, as_gray=True)
    img = resize(img, (IMG_SIZE, IMG_SIZE), preserve_range=True)  # ต้องเหมือนตอนเทรนเป๊ะ
    img = img / 255.0
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


def predict_folder(folder_path, output_csv="out.csv"):
    results = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith('.jpg'):
            path = os.path.join(folder_path, fname)
            pred_class, conf = predict(path)
            results.append((fname, pred_class, conf))
            print(f"{fname} -> {pred_class} ({conf:.2f}%)")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_class", "confidence(%)"])
        writer.writerows(results)
    print(f"\nบันทึกผลลัพธ์ทั้งหมดไว้ที่ {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("วิธีใช้: python TestingCNN.py <path_ไฟล์ภาพ_หรือ_โฟลเดอร์>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        predict_folder(target)
    else:
        pred_class, conf = predict(target)
        print(f"Predicted: {pred_class} ({conf:.2f}%)")
