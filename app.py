import os
import cv2
import matplotlib.pyplot as plt
from your_apple_module import (
    load_dataset,
    train_model,
    evaluate_model,
    predict_image,
    GRADE_MAPPING,
    BASE_PRICE
)

# 設定 sample 資料夾
SAMPLE_FOLDER = 'sample apple'

# Step 1: 載入並準備資料
print("📥 載入資料集中...")
X, y = load_dataset()
print(f"✅ 載入完成，共 {len(X)} 筆樣本")

if len(X) == 0:
    print("❌ 錯誤：找不到有效資料，請檢查資料夾結構")
    exit()

# Step 2: 模型訓練與測試
print("🛠️ 開始訓練模型...")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = train_model(X_train, y_train)

print("\n📊 模型評估：")
evaluate_model(model, X_test, y_test)

# Step 3: 預測 sample apple 中的圖片
if not os.path.exists(SAMPLE_FOLDER):
    print(f"❌ 找不到資料夾 {SAMPLE_FOLDER}")
    exit()

sample_files = [f for f in os.listdir(SAMPLE_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
if not sample_files:
    print("ℹ️ sample apple 資料夾中沒有圖片")
    exit()

print("\n🔍 預測 sample apple 中的圖片：")
plt.figure(figsize=(12, 6))
for i, file in enumerate(sample_files):
    path = os.path.join(SAMPLE_FOLDER, file)
    img = cv2.imread(path)
    if img is None:
        continue

    grade, price = predict_image(model, img)
    print(f"\n📷 圖片: {file}")
    print(f"🍎 分級結果：{grade}")
    print(f"💰 建議售價：{price:.1f} 元（基準價 {BASE_PRICE}）")

    # 顯示圖片與分級
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, len(sample_files), i+1)
    plt.imshow(img_rgb)
    plt.title(f"{grade}\n{price:.1f} 元", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()
