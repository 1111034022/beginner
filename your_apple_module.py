import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 分類與價格設定
CATEGORIES = [
    'U.S. Fancy apples',
    'U.S. No.1 apples',
    'U.S. No.2 apples',
    'Inedible (or process) apples'
]
GRADE_MAPPING = {
    0: 'U.S. Fancy',
    1: 'U.S. No.1',
    2: 'U.S. No.2',
    3: 'Cider'
}
PRICE_RATIOS = {
    'U.S. Fancy': 0.9,
    'U.S. No.1': 0.6,
    'U.S. No.2': 0.3,
    'Cider': 0.1
}
BASE_PRICE = 56.3


def extract_features(image):
    """從圖片提取特徵（亮度、顏色、暗斑比例）"""
    image = cv2.resize(image, (100, 100))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    brightness = np.mean(hsv[:, :, 2])
    darkness_ratio = np.sum(hsv[:, :, 2] < 50) / (100 * 100)
    green_ratio = np.sum((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85)) / (100 * 100)

    return np.array([brightness, darkness_ratio, green_ratio])


def load_dataset(base_path='./'):
    """載入訓練資料並標註分類"""
    features, labels = [], []
    for idx, category in enumerate(CATEGORIES):
        folder = os.path.join(base_path, category)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is None:
                continue
            feature = extract_features(img)
            features.append(feature)
            labels.append(idx)
    return np.array(features), np.array(labels)


def train_model(X, y):
    """訓練 SVM 模型"""
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    return model


def predict_image(model, img):
    """對單一圖片進行預測，返回分類與售價"""
    features = extract_features(img).reshape(1, -1)
    pred_idx = model.predict(features)[0]
    grade = GRADE_MAPPING[pred_idx]
    price = BASE_PRICE * PRICE_RATIOS[grade]
    return grade, price


def evaluate_model(model, X_test, y_test):
    """列印模型準確率與報告"""
    y_pred = model.predict(X_test)
    print("📊 模型準確率：", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=list(GRADE_MAPPING.values())))
