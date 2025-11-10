# ======================================
# 1. 라이브러리 임포트
# ======================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight  # 👈 추가됨

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import Input




# ======================================
# 2. 데이터 불러오기
# ======================================
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)
print(df.shape)

# ======================================
# 3. 입력/출력 분리 및 스케일링
# ======================================
X = df.drop(columns=['Class'])
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================
# 4. Train/Test Split
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ======================================
# 5. 클래스 가중치 계산
# ======================================
# 클래스별 샘플 수 기반으로 자동 계산
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# 딕셔너리 형태로 변환
class_weights = {0: class_weights[0], 1: class_weights[1]}
print("Class Weights:", class_weights)
# 예: {0: 0.5, 1: 289} 처럼 나올 수 있음

# ======================================
# 6. MLP 모델 정의
# ======================================
def build_mlp(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),    # 👈 권장 방식
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model

# ======================================
# 7. 라벨 희소 실험 (class_weight 적용)
# ======================================
ratios = [1.0, 0.5, 0.1]
results = []

for ratio in ratios:
    print(f"\n============================")
    print(f"Label Ratio = {int(ratio*100)}%")
    print(f"============================")

    # 비율만큼의 학습 데이터만 사용
    n_samples = int(len(X_train) * ratio)
    X_sub = X_train[:n_samples]
    y_sub = y_train[:n_samples]

    # 모델 생성 및 학습
    model = build_mlp(X_sub.shape[1])
    history = model.fit(
        X_sub, y_sub,
        epochs=8,
        batch_size=512,
        validation_split=0.2,
        verbose=1,
        class_weight=class_weights   # 👈 핵심: 클래스 가중치 적용
    )

    # 평가
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    auc = roc_auc_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)

    print(f"Ratio {int(ratio*100)}% → AUC: {auc:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    results.append({
        "Ratio": ratio,
        "AUC": auc,
        "Recall": recall,
        "F1": f1
    })

# ======================================
# 8. 결과 시각화
# ======================================
results_df = pd.DataFrame(results)
print("\n\n실험 결과 요약:")
print(results_df)

plt.figure(figsize=(7,5))
plt.plot(results_df["Ratio"]*100, results_df["AUC"], marker='o', label='AUC')
plt.plot(results_df["Ratio"]*100, results_df["Recall"], marker='s', label='Recall')
plt.plot(results_df["Ratio"]*100, results_df["F1"], marker='^', label='F1-score')
plt.title("Label Ratio vs Model Performance (with Class Weights)")
plt.xlabel("Labeled Data Ratio (%)")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.show()

