import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
plt.rcParams['font.family'] = 'AppleGothic'  # 한글 폰트

# 데이터 불러오기
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv('airline_passengers.csv', parse_dates=['Month'], index_col='Month')
print(df.head())
print(df.shape)  # (144, 1)

## 2단계 시계열 데이터에서 이상치 자동 탐지하기
# 2.1 특성 엔지니어링 ( 시계열 맞추기 )
# 2.2 Isoilation Forest 작동
# 2.3 결과

# ===== 2단계: Isolation Forest 이상치 탐지 =====
print("\n=== Isolation Forest 시작 ===")

# 시계열 특성 생성
df['lag1'] = df['Passengers'].shift(1)           # 이전 월 승객수
df['rolling_mean'] = df['Passengers'].rolling(3).mean()  # 3개월 이동평균
df.dropna(inplace=True)

# Isolation Forest (5% 이상치 가정)
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso_forest.fit_predict(df[['Passengers', 'lag1', 'rolling_mean']])

# 결과 출력
anomaly_count = (df.anomaly == -1).sum()
print(f"✅ 이상치 탐지 완료! 이상치 {anomaly_count}개")
print("\n이상치 목록:")
print(df[df.anomaly == -1][['Passengers']])

print("\n=== 2단계 완료! 3단계 선형회귀 대기 중 ===")

# ===== 3단계: 이상치 제거 후 선형 회귀 예측 =====
print("\n=== 3단계: 선형 회귀 예측 시작 ===")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# 1) 정상 데이터만 학습에 사용 (anomaly == 1)
normal = df[df['anomaly'] == 1].copy()
normal['t'] = np.arange(len(normal))  # 시간 인덱스(0,1,2,...)

X_train = normal[['t']]
y_train = normal['Passengers']

model = LinearRegression()
model.fit(X_train, y_train)

# 2) 전체 구간 예측
df['t'] = np.arange(len(df))
y_pred = model.predict(df[['t']])

# 3) 성능 확인
mse = mean_squared_error(df['Passengers'], y_pred)
print(f"예측 MSE: {mse:.2f}")

# 4) 그래프 두 개로 비교
plt.figure(figsize=(14, 8))

# (1) 원본 + 이상치
plt.subplot(2, 1, 1)
plt.plot(df.index, df['Passengers'], 'b-', label='실제 값', linewidth=2)
plt.scatter(
    df[df['anomaly'] == -1].index,
    df[df['anomaly'] == -1]['Passengers'],
    color='red',
    s=120,
    label='이상치',
    zorder=5,
)
plt.title('1. 원본 데이터와 이상치', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# (2) 선형회귀 예측
plt.subplot(2, 1, 2)
plt.plot(df.index, df['Passengers'], 'b-', label='실제 값', linewidth=2, alpha=0.7)
plt.plot(df.index, y_pred, 'r--', label='선형 회귀 예측', linewidth=3)
plt.scatter(
    df[df['anomaly'] == -1].index,
    df[df['anomaly'] == -1]['Passengers'],
    color='red',
    s=120,
    label='이상치',
    zorder=5,
)
plt.title('2. 이상치 제거 후 선형 회귀 예측', fontweight='bold')
plt.xlabel('날짜')
plt.ylabel('승객 수')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== 3단계 완료: 이상치 탐지 + 예측 파이프라인 완성 ===")
print(f"예측 MSE: {mse:.2f}")