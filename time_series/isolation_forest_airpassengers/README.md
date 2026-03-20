# Time Series Anomaly Detection with Isolation Forest

## 1. 프로젝트 개요
- 데이터: AirPassengers (1949–1960, 월별 국제 항공 승객 수, 144개 관측치)
- 목표:
  - 시계열 데이터에서 Isolation Forest로 이상치 탐지
  - 이상치 정보를 활용해 선형 회귀로 추세 예측

## 2. 사용한 기술 스택
- Python 3
- pandas, numpy
- scikit-learn (IsolationForest, LinearRegression)
- matplotlib

## 3. 분석 단계

### 3.1 데이터 로드 및 전처리
- `airline_passengers.csv`를 `Month` 컬럼을 시계열 인덱스로 하여 로드.
- 데이터 크기: (144, 1)
- 시각화를 통해 전반적인 추세(상승)와 계절성(연간 패턴) 확인.

### 3.2 시계열 이상치 탐지 (Isolation Forest)
- 시계열 특성 피처 생성:
  - `lag1`: 한 달 전 승객 수
  - `rolling_mean`: 3개월 이동 평균
- Isolation Forest 설정:
  - `contamination=0.05` (상위 5%를 이상치로 가정)
  - `random_state=42`
- 결과:
  - 이상치 개수: 8개
  - 주로 피크가 비정상적으로 크거나 패턴에서 벗어난 구간에서 발생.
- 시각화:
  - 파란 선: 전체 시계열
  - 빨간 점: Isolation Forest가 탐지한 이상치

### 3.3 이상치 제거 후 선형 회귀 예측
- `anomaly == 1`인 정상 구간만 학습 데이터로 사용.
- 시간 인덱스 `t = 0, 1, 2, ...`를 피처로 사용하여 Linear Regression 학습.
- 전체 기간에 대한 예측값을 생성하고 실제 값과 비교.
- 성능 지표:
  - MSE (Mean Squared Error): 약 2173.27
- 해석:
  - 선형 회귀는 장기적인 상승 추세는 잘 설명하지만,
  - 계절성(연간 주기)은 반영하지 못해 오차가 남음.
  - 향후 ARIMA, SARIMA, LSTM 등 시계열 전용 모델로 확장 가능.

## 4. 파일 설명
- `Isolation forest.py`  
  - 데이터 로드 → 이상치 탐지 → 선형 회귀 예측 전 과정을 포함한 메인 스크립트.
- `airline_passengers.csv`  
  - AirPassengers 원본 데이터셋.

## 5. 실행 방법
```bash
cd time_series/isolation_forest_airpassengers
python3 "Isolation forest.py"
