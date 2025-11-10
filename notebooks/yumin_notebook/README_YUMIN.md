# Yumin Experiment Notebook

> 작성자: 김유민  
> 프로젝트: SSL-FinFraud-Detection  
> 주제: Self-Supervised Learning(SSL) 사전학습과 라벨 비율별 실험을 통한 데이터 효율성 비교

---

## 1. 개요

이 노트북은 기존의 신용카드 사기 탐지 모델에 **자기지도학습(Self-Supervised Learning)** 을 적용해  
라벨이 부족한 상황에서도 모델이 스스로 데이터의 패턴을 학습할 수 있도록 하는 실험을 다룹니다.  

또한 **라벨 데이터 비율(10%, 20%, 50%, 100%)** 을 조절해 학습에 얼마나 영향을 주는지도 확인했습니다.

---

## 2. 실험 구성

### (1) SSL Pretraining - Denoising Autoencoder

- 입력 데이터의 일부를 가려서(마스킹) 원래 데이터를 복원하도록 학습합니다.  
- 이를 통해 **라벨 없이도 데이터의 특징을 잘 잡는 Encoder**를 만듭니다.

```python
dae = TabularDAE(n_features, hidden=128, bottleneck=32)
recon, z = dae(x_masked)
loss = MSE(recon, x)
```

훈련이 끝나면 `dae.encoder()` 부분을 사용해  
새로운 입력 특징(`Z_train`, `Z_test`)을 만듭니다.

---

### (2) Downstream Classifier

- 사전학습으로 만든 Encoder의 출력을 입력으로 사용합니다.  
- 두 가지 분류기를 사용했습니다:
  1. Logistic Regression  
  2. MLP Classifier (2개 은닉층)

- 성능 평가는 **AUC**, **F1-score**, **Accuracy** 로 진행했습니다.

```python
clf_lr = LogisticRegression(max_iter=2000, class_weight="balanced")
clf_lr.fit(Z_train, y_train)
auc = roc_auc_score(y_test, clf_lr.predict_proba(Z_test)[:,1])
```

---

### (3) Label Ratio Experiment

- 학습용 라벨 데이터를 10%, 20%, 50%, 100%로 줄여가며 성능을 비교했습니다.  
- 라벨이 적을수록 성능이 얼마나 떨어지는지,  
  그리고 SSL 사전학습이 있을 때 얼마나 버티는지를 확인했습니다.

```python
ratios = [0.1, 0.2, 0.5, 1.0]
for r in ratios:
    sss = StratifiedShuffleSplit(train_size=r)
    idx = next(sss.split(Z_train, y_train))[0]
    clf.fit(Z_train[idx], y_train[idx])
```

#### 예시 결과

| label_ratio | AUC | F1 | Acc | n_train |
|--------------|-----|----|------|----------|
| 0.1 | 0.73 | 0.61 | 0.68 | 2400 |
| 0.2 | 0.78 | 0.65 | 0.70 | 4800 |
| 0.5 | 0.82 | 0.71 | 0.75 | 12000 |
| 1.0 | 0.85 | 0.74 | 0.78 | 24000 |

> **결론:** SSL로 학습한 Encoder를 사용하면  
> 라벨이 적어도 성능 하락이 덜하고 더 안정적으로 작동합니다.

---

## 3. 요약

| 항목 | 설명 |
|------|------|
| SSL Pretraining | Denoising Autoencoder로 Encoder 학습 |
| Downstream | Encoder 출력으로 분류기(Logistic, MLP) 학습 |
| Label Ratio Experiment | 라벨 비율(10~100%)별 성능 비교 |
| 결론 | SSL 모델이 적은 라벨에서도 더 높은 AUC 유지 |

---

## 4. 실행 환경

| 항목 | 버전 |
|------|------|
| Python | 3.10+ |
| PyTorch | 2.0 이상 |
| scikit-learn | 1.2 이상 |
| pandas, numpy | 최신 버전 |
