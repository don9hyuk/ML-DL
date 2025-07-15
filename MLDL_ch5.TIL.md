# Ch.5 트리 알고리즘

# 5-1. 결정 트리

- 데이터를 **질문을 던지며 나누어 가는** 방식으로 예측하는 지도학습 모델
- 마치 스무고개처럼 조건을 기준으로 데이터를 분할하면서 최종 결론(예측값)에 도달

```jsx
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target)) # 훈련 세트
print(dt.score(test_scaled, test_target)) # 테스트 세트
```

- 훈련 세트에 대한 점수가 엄청 높은 반면  테스트 세트의 성능은 조금 낮음 → 과대적합된 모델

## 결정 트리

```jsx
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()
```

![image.png](image.png)

- 학습된 결정 트리 모델을 트리 구조로 시각화
- 맨 위의 노드: 루트 노드 / 맨 아래 노드: 리프 노드
- 노드는 결정 트리를 구성하는 핵심 요소이며 훈련 데이터의 특성에 대한 테스트를 표현

```jsx
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol',
'sugar', 'pH'])
plt.show()
```

![image.png](image%201.png)

- 트리의 깊이를 제한해서 출력
- 루트 노드는 당도가-0.239 이하인지 질문
- 만약 어떤 샘플의 당도가-0.239와 같거나 작으면 왼쪽 가지로, 그렇지 않으면 오른쪽 가지로 이동

       → 왼쪽이 Yes, 오른쪽이 No 

- 리프 노드에서 가장 많은 클래스가 예측 클래스가 됨
- 두 노드 모두 양성 클래스의 개수가 많기 때문에 여기서 멈춘다면

       왼쪽과 오른쪽 노드의 샘플 모두 양성 클래스로 예측

## 불순도

### 지니 불순도 (gini)

![KakaoTalk_20250715_101547598.png](KakaoTalk_20250715_101547598.png)

- Pk: 클래스 kk의 비율
- 이 값이 낮을수록 노드가 순수함 (한 클래스에 몰려있음)
- `DecisionTreeClassifier(criterion='gini')`에서 기본 사용됨

지니 불순도 = 1 - (음성 클래스 비율의 제곱 + 양성 클래스 비율의 제곱)

### 정보 이득

![KakaoTalk_20250715_101955514.png](KakaoTalk_20250715_101955514.png)

- 부모와 자식 노드 사이의 불순도 차이

### 엔트로피 불순도

![KakaoTalk_20250715_102119843.png](KakaoTalk_20250715_102119843.png)

- 클래스 비율에 로그를 곱해서 측정
- `DecisionTreeClassifier(criterion='entropy')`로 사용 가능
- 노드의 클래스 비율을 사용하지만 지니 불순도처럼 제곱이 아니라 밑이 2인 로그를 사용하여 곱함
- 보통 기본값인 지니 불순도와 엔트로피 불순도가 만든 결과의 차이는 크지 않음

### 정리 - 결정트리 알고리즘

- 불순도 기준을 사용해 정보 이득이 최대가 되도록 노드를 분할
- 노드를 순수하게 나눌수록 정보 이득이 커짐
- 새로운 샘플에 대해 예측할 때 노드의 질문에 따라 트리를 이동
- 마지막에 도달한 노드의 클래스 비율을 보고 예측

## 가지치기

- 트리의 복잡도를 줄여 일반화 성능을 향상시키는 것
- 과수원에서 열매를 더 잘 맺게 하려고 가지를 잘라주는 것과 동일
- 가장 쉬운 가지치기 방법: 트리의 최대 깊이 제한

```jsx
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
```

- 루트 노드 아래로 최대 3개의 노드까지만 성장하도록
- 훈련 세트의 성능은 낮아졌지만 테스트 세트의 성능은 거의 그대로

```jsx
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```

![image.png](image%202.png)

### 노드 분할 기준

- 루트 노드: 당도(sugar)를 기준으로 훈련 세트를 나눔
- 깊이 1의 노드들:
    - 왼쪽: 당도 기준
    - 오른쪽 일부: 알코올 도수(alcohol) 또는 pH 사용
- 깊이 3의 노드들: 리프 노드(최종 예측 노드)

### 해석 예시

- 세 번째 리프 노드는 유일하게 **레드 와인(음성 클래스)**이 더 많음
- 이 노드에 도달하려면:
    - 당도는 0.802 < 당도 ≤ -0.239
    - 알코올 도수는 ≤ 0.454

즉, 당도와 알코올 도수 조건을 모두 만족하는 와인은 레드 와인으로 분류됨

```jsx
dt = DecisionTreeClassifier(max_depth=3,random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))
```

- 전처리하기 전의 데이터셋을 다시 흔련한 결과 점수가 정확히 같음

### 특성 중요도

- 결정 트리는 어떤 특성이 가장 유용한지 나타내는 특성 중요도를 계산
- 결정 트리 모델의 feature_importances_속성에 저장

```jsx
print(dt.feature_importances_)
```

- [0.12345626  0.86862934  0.0079144 ]
- 두 번째 특성인 당도가 0.87 정도로 특성 중요도가 가장 높음
- 값을 모두 더하면 1
- 각 노드의 정보 이득과 전체 샘플에 대한 비율을 곱한 후 특성 별로 더하여 계산
- 특성 중요도를 활용하면 결정 결정 트리의 특성 중 트리 모델을 특성 선택에 활용 가능

# 5-2. 교차 검증과 그리드 서치

### 테스트 세트

- 테스트 세트는 모델의 일반화 성능을 평가하기 위한 용도
- "이 모델을 실전에 투입하면 이 정도 성능이 나올 것이다"를 가늠하는 데 사용

남용의 문제

- 테스트 세트를 여러 번 사용하면, 모델이 테스트 세트에 과적합될 수 있음
- 그 결과, 테스트 세트에서 얻은 성능이 실제 일반화 성능을 과대평가할 수 있음

올바른 사용

- 테스트 세트는 최종 평가 때 단 한 번만 사용하는 것이 바람직
- 모델 개발 및 하이퍼파라미터 튜닝 과정에서는 테스트 세트를 사용하지 말아야 함

### 검증 세트

- 테스트 세트를 사용하지 않고 측정하는 간단한 방법: 훈련 세트를 또 나누기
- 예) 전체 데이터 중 20%는 테스트 세트 / 나머지 80%는 훈련 세트

              이 훈련 세트 중에서 다시 20%는 내어 검증 세트

- **훈련 세트로 모델을 학습**
    - 모델은 여기서 패턴을 배움
- **검증 세트로 모델을 평가**
    - 여러 **하이퍼파라미터(예: max_depth)**를 바꿔가며 성능을 비교
    - 가장 성능이 좋은 설정을 선택
- *선택한 하이퍼파라미터로 전체 훈련 데이터(훈련 + 검증)**를 사용해 모델 재훈련
- **테스트 세트로 최종 성능 평가**
    - 이 점수가 **실제 실전 투입 시 기대할 수 있는 성능**을 가장 잘 나타냄

```jsx
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

print(sub_input.shape, val_input.shape)
```

- train_test_split() 함수 2번 적용해 훈련 세트와 검증 세트로 나누기
- 훈련 세트: 5197개 → 4157개 / 검증 세트: 1040개

```jsx
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))
```

- 0.9971133028626413
0.864423076923077
- 과대적합된 상태의 모델

### 교차 검증

- 검증 세트를 따로 떼면 훈련 데이터 양이 줄어들고 너무 작게 떼면 검증 점수가 불안정해질 수 있음

교차 검증

- 데이터를 여러 조각으로 나눈 뒤,
- 각 조각을 한 번씩 검증 세트로 사용하고 나머지는 훈련 세트로 사용
- 이 과정을 반복하여 검증 점수를 평균하면 보다 안정적인 성능 평가 가능
- 장점
1. 훈련에 더 많은 데이터를 사용할 수 있음
2. 검증 점수가 보다 신뢰할 수 있음
3. 과적합 방지와 모델 선택의 안정성 향상

![KakaoTalk_20250715_112008776.png](KakaoTalk_20250715_112008776.png)

```jsx
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)

import numpy as np
print(np.mean(scores['test_score']))
```

- 교차 검증을 수행하면 입력한 모델에서 얻을 수 있는 최상의 검증 점수를 가늠해 볼 수 있음

```jsx
from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))
```

```jsx
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))
```

- 훈련 세트를 섞은 후 10-폴드 교차 검증을 수행

### 하이퍼파라미터 튜닝

| **모델 파라미터** | 모델이 **데이터로부터 학습**하는 값 | 선형 회귀의 기울기(가중치), 결정 트리의 분할 기준 |
| --- | --- | --- |
| **하이퍼파라미터** | 사용자가 **직접 설정해야 하는 값** | `max_depth`, `min_samples_split` 등 |
1. 라이브러리의 **기본값(default)** 으로 모델 훈련
2. **검증 세트 점수** 또는 **교차 검증 결과**를 통해 성능 평가
3. 하이퍼파라미터를 **조금씩 바꿔가며** 모델 재훈련 및 평가
    
    → 반복적인 시도 필요
    
- 하이퍼파라미터 간의 상호작용 존재
- 하나의 하이퍼파라미터를 고정한 상태에서 다른 것을 튜닝하면  최적의 조합을 놓칠 수 있음
- 여러 개를 동시에 조합해서 튜닝해야 함→ GridSearchCV 또는 RandomizedSearchCV 사용
- 사이킷런의 **그리드 서치** 사용

```jsx
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params,n_jobs=-1)

gs.fit(train_input, train_target)

dt = gs.best_estimator_
print(dt.score(train_input, train_target))
```

- 그리드 서치는 훈련이 끝나면 25개의 모델 중에서 검증 점수가 가장 높은 모델의 매개변수 조합으로

       전체 훈련 세트에서 자동으로 다시 모델을 훈련

- 그리드 서치로 찾은 최적의 매개변수는 best_params 속성에 저장

```jsx
print(gs.best_params_)

print(gs.cv_results_['mean_test_score'])

best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])
```

1. 먼저 탐색할 매개변수를 지정
2. 훈련 세트에서 그리드 서치를 수행하여 최상의 평균 검증 점수가 나오는 매개변수 조합
을 찾음 →이 조합은 그리드 서치 객체에 저장
3. 그리드 서치는 최상의 매개변수에서 전체 훈련 세트를 사용해 최종 모델을 훈련,

       이 모델도 그리드 서치 객체에 저장됨

```jsx
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
'max_depth': range(5, 20, 1),
'min_samples_split': range(2, 100, 10)}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

print (gs.best_params_)

print (np.max(gs.cv_results_['mean_test_score']))
```

- 하이퍼파라미터 탐색 범위 설정, 총 조합 수는 1350개
- 모든 조합을 시험하여 최적의 조합을 찾고 모델 학습과 교차 검증을 수행
- 가장 높은 성능을 낸 최적의 하이펴파라미터 조합 출력
- 가장 높은 평균 교차 검증 점수 출력

### 랜덤 서치

- 매개변수의 값이 수치일 때 값의 범위나 간격을 미리 정하기 어렵거나

       너무 많은 매개 변수 조건이 있어 그리드 서치 시간이 오래 걸릴 때

- 매개변수 값의 목록을 전달하는 것이 아니라 매개변수를 샘플링할 수 있는 확률 분포 객체를 전달

```jsx
from scipy.stats import uniform, randint

rgen = randint(0, 10)
rgen.rvs(10)

np.unique(rgen.rvs(1000), return_counts=True)

ugen = uniform(0, 1)
ugen.rvs(10)
```

- 균등분포 샘플링: 0에서 10 사이의 범위를 갖는 randint 객체를 만들고 10개의 숫자를 샘플링
- 샘플링 횟수는 시스템 자원이 허락하는 범위 내에서 최대한 크게

```jsx
params = {'min_impurity_decrease': uniform(0.0001,0.001),
'max_depth': randint(20, 50),
'min_samples_split': randint(2, 25),
'min_samples_leaf': randint(1, 25),
}
```

```jsx
from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params,
n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)

print(gs.best_params_)

print (np.max(gs.cv_results_['mean_test_score']))

 dt = gs.best_estimator_
print(dt.score(test_input, test_target))
```

- 탐색할 매개변수 범위 설정 후
- 0.0001에서 0.001 사이의 실숫값,  20에서 50 사이의 정수, 2에서 25 사이의 정수,  1에서 25 사이의 정수를 샘플링
- params에 정의된 매개변수 범위에서 총 100번 (n_iter 매개변수)을 샘플링하여

       교차 검증을 수행하고 최적의 매개변수 조합을 탐색

- 최적의 매개변수 조합과 최고의 교차 검증 점수 출력
- 테스트 세트 점수는 검증 세트에 대한 점수보다 조금 작은 것이 일반적

# 5-3. 트리의 앙상블

### 정형 데이터와 비정형 데이터

**정형 데이터**

- 행과 열로 구성된 표 형태의 데이터, 정해진 형식과 구조를 갖춘 데이터

       → 엑셀, 테이블, 관계형 데이터베이스(DB) 등에 저장 가능한 데이터

- 엑셀 시트, CSV 파일
- 고객 정보: 이름, 나이, 주소, 가입일 등
- 판매 내역: 상품명, 수량, 가격, 날짜 등
- 센서 데이터: 시간, 측정값, 위치 등

**비정형 데이터**

- 고정된 형식 없이 저장된 데이터. 행과 열로 쉽게 표현되지 않음.
- 텍스트: 뉴스 기사, 블로그, SNS 글 / 이미지: 사진, X-ray, 위성 사진
- 음성: 녹음 파일, 음성 인식 입력 / 영상: CCTV 영상, 유튜브 영상
- 문서 파일: PDF, 워드 파일
- 분석 전 전처리와 구조화 과정이 필요
- 딥러닝(NLP, CV) 기술이 자주 활용됨

**앙상블 학습**

- 정형 데이터를 다루는 데 가장 뛰어난 성과를 내는 알고리즘
- 여러 개의 머신러닝 모델을 결합하여 성능을 향상시키는 기법

### 랜덤 포레스트

- 앙상블 학습의 대표 주자 중 하나로 안정적인 성능이 특징
- 결정 트리를 랜덤하게 만들어 결정 트리(나무)의 숲을 구성 후,  각 결정 트리의 예측을 사용해 최종 예측

**부트스트랩 샘플링**

- 훈련 데이터를 중복 허용하여 샘플링
- 예: 1,000개 중 1,000개를 랜덤하게 중복 포함해서 뽑음
- 각 결정 트리는 이렇게 생성된 다른 훈련 세트로 학습됨

**특성 무작위 선택**

- 각 노드 분할 시, 전체 특성 중 일부만 랜덤하게 선택하여 최적의 분할 기준 탐색
- 분류 모델 (`RandomForestClassifier`):
    
    → 기본적으로 전체 특성 수의 √개를 사용
    
- 회귀 모델 (`RandomForestRegressor`):
    
    → 전체 특성 모두 사용
    

**예측 방식**

- 분류:
    
    → 모든 트리의 예측 확률 평균, 가장 높은 확률의 클래스를 최종 예측
    
- 회귀:
    
    → 모든 트리의 예측 값 평균
    

**장점**

- 과적합 방지: 데이터와 특성 모두에 randomness가 적용되어 과적합 감소
- 일반화 성능 우수: 테스트 세트에서 성능이 안정적
- 매개변수 튜닝 없이도 기본 설정만으로 좋은 성능 가능

```jsx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(
data, target, test_size=0.2, random_state=42)

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target,
return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

```

- 교차 검증을 수행한 후 훈련 세트와 검증 세트의 점수를 비교하며 과대적합을 파악

```jsx
rf.fit(train_input, train_target)
print(rf.feature_importances_)
```

- 랜덤 포레스트 모델을 훈련 세트에 훈련한 후 특성 중요도를 출력
- 랜덤 포레스트의 특성 중요도는 각 결정 트리의 특성 중요도를 취합한 것
- 하나의 특성에 집중하지 않고 더 많은 특성에 적용 → 과대적합을 줄이고 일반화 성능을 높임

**OOB 점수**

- 랜덤 포레스트에서 부트스트랩 샘플링을 할 때 뽑히지 않은 샘플
- 각 결정 트리는 자기 훈련에 쓰이지 않은 OOB 샘플로 자기 성능을 평가할 수 있음
- 모든 트리에 대해 OOB 점수를 평균 내면 전체 모델의 검증 점수와 유사한 성능 추정치가 됨

### 엑스트라 트리

- 랜덤 포레스트와 구조는 매우 유사 → 여러 결정 트리를 앙상블하여 예측
- 무작위성이 커서 개별 트리 성능은 약할 수 있음
- 그러나 많은 트리를 앙상블하면 과적합을 억제하고 성능이 좋아짐
- 계산 속도 빠르고, 일반화 성능 안정적

| 항목 | 랜덤 포레스트 | 엑스트라 트리 |
| --- | --- | --- |
| 샘플링 방식 | **부트스트랩 샘플** 사용 (중복 허용 샘플링) | **전체 훈련 세트 사용** (샘플링 없음) |
| 분할 기준 | **최적의 분할**을 찾음 | **완전히 랜덤한 값**으로 분할 |
| 무작위성 | 상대적으로 낮음 | 더 높음 |
| 계산 속도 | 상대적으로 느림 | **더 빠름** |
| 과대적합 위험 | 있음 | **더 낮음** (무작위성이 많기 때문) |

```jsx
 from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target,
return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

et.fit(train_input, train_target)
print(et.feature_importances_)
```

### 그레이디언트 부스팅

- 얕은 결정 트리(보통 깊이 3)**를 **여러 개 이어붙이는 방식**의 앙상블 학습
- 이전 트리가 만든 **오차를 보완**하는 새로운 트리를 계속 추가함

**작동 원리**

1. 처음에는 간단한 트리로 시작
2. 그 트리의 예측이 틀린 부분(오차)에 집중하여 **다음 트리 학습**
3. 이렇게 오차를 계속 줄여가며 **트리를 순차적으로 추가**
4. 마지막에 **모든 트리의 예측을 더하여 최종 예측**

- 트리를 추가할 때 **경사 하강법(gradient descent)**을 사용
- 즉, **손실 함수가 최소가 되는 방향**으로 **모델을 조금씩 개선**
- **분류**: 로지스틱 손실 함수
    
    **회귀**: 평균 제곱 오차(MSE)
    
- 결정 트리의 개수를 늘려도 과대적합에 매우 강함

```jsx
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target,
return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2,
random_state=42)
scores = cross_validate(gb, train_input, train_target,
return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

gb.fit(train_input, train_target)
print(gb.feature_importances_)
```

### 히스토그램 기반 그레이디언트 부스팅

- 정형 데이터를 다루는 머신러닝 알고리즘 중에 가장 인기가 높은 알고리즘
- 연속적인 특성 값을 구간(bin) 으로 나눠 히스토그램을 생성한 후,
    
    이 구간 단위로 분할 후보를 탐색하여 트리를 학습
    
- → 계산량과 메모리 사용량을 대폭 줄임

| 항목 | **기존 Gradient Boosting** (`GradientBoostingClassifier`) | **히스토그램 기반 Gradient Boosting** (`HistGradientBoostingClassifier`) |
| --- | --- | --- |
| **분할 방식** | 모든 가능한 **연속값** 탐색 | 데이터를 **구간(bin)**으로 나눠 **대표값 기준** 탐색 |
| **속도** | 느림 (수치형 특성 많을수록) | **훨씬 빠름** (구간 단위 탐색) |
| **메모리 사용량** | 큼 (전체 데이터를 그대로 사용) | 작음 (히스토그램만 저장) |
| **대규모 데이터** | 느리고 무거움 | **매우 효율적** |
| **정확도** | 높음 | 비슷하거나 더 높음 (특히 큰 데이터에서) |
| **범주형 특성 처리** | 수동 인코딩 필요 | 자동 처리 가능 (`categorical_features` 지원) |
| **사용 가능 시점** | 오래전부터 지원 | **사이킷런 0.22부터 추가** |
| **병렬 처리** | 제한적 | 지원 (내부적으로 빠름) |

```jsx
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target,
return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

```jsx
from sklearn.inspection import permutation_importance
hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target,
n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)

 result = permutation_importance(hgb, test_input, test_target,
n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)

hgb.score(test_input, test_target)
```

- 앙상블 모델은 확실히 단일 결정 트리보다 좋은 결과를 얻을 수 있음
- 가장 대표적인 라이브러리 - **XGBoost**

```jsx
from xgboost import XGBClassifier
xgb = XGBClassifier(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target,
return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

- LightGBM

```jsx
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target,
return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```
