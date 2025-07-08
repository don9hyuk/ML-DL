# Ch.4 다양한 분류 알고리즘

# 4-1. 로지스틱 회귀

### 럭키백의 확률

- 7개 생선 종류 각각에 대한 확률
- 다중 클래스 확률 분류 문제 → k-최근접 이웃(k-NN) 분류기를 확률 추정기로 사용

데이터 준비

```jsx
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()

print (pd.unique(fish['Species']))
```

종: ['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch', 'Pike', 'Smelt']

```jsx
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_input[:5])
```

Species 열을 빼고 나머지 5개 열을 선택해 fish_input에 저장

```jsx
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
fish_input, fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler ()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

```

- 데이터를 훈련 세트와 테스트 세트로 나눔
- 사이킷런의 StandardScaler 클래스를 사용해 훈련 세트와 테스트 세트를 표준화 전처리

### k-최근접 이웃 분류기의 확률 예측

```jsx
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
```

- 최근접 이웃 개수인 k를 3으로 지정하여 예측
- 타깃 데이터를 만들 때 fish [Species]를 사용해 만들었기 때문에

       훈련 세트와 테스트 세트의 타깃 데이터에도 7개의 생선 종류 존재 → **다중 분류**

```jsx
print(kn.classes_)
```

['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']

```jsx
 print(kn.predict(test_scaled[:5]))
```

['Perch' 'Smelt' 'Pike' 'Perch' 'Perch']

- 타깃값을 그대로 사이킷런 모델에 전달하면 순서가 자동으로 알파벳 순으로 매겨짐

       → 처음 입력한 순서와 다름

```jsx
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4)) 
```

[[0.     0.     1.     0.     0.     0.     0.    ]
[0.     0.     0.     0.     0.     1.     0.    ]
[0.     0.     0.     1.     0.     0.     0.    ]
[0.     0.     0.6667 0.     0.3333 0.     0.    ]
[0.     0.     0.6667 0.     0.3333 0.     0.    ]]

- 사이킷런의 분류 모델은 predict proba () 메서드로 클래스별 확률값을 반환
- 테스트 세트에 있는 처음 5개의 샘플에 대한 확률 출력 결과

       첫 번째 열이 'Bream'에 대한 확률, 두 번째 열이 'Parkki'에 대한 확률

- 3개의 최근접 이웃을 사용하기 때문에 가능한 확률은 0/3, 1/3, 2/3, 3/3이 전부

### 로지스틱 회귀

- 선형 회귀와 동일하게 선형 방정식을 학습
- z = a x (Weight) + b x (Length) + c x (Diagonal) + d x (Height) + e x (Width) + f
- z 는 어떤 값도 가능하지만 확률이 되려면 0~1 사이의 값이 되어야함

**시그모이드 함수 (로지스틱 함수)**

- z가 아주 큰 음수일 때 0이 되고, z가 아주 큰 양수일 때 1이 되도록 바꾸는 방법

![image.png](6143e3e2-ca3e-40fe-9c38-4ac78121b4f2.png)

```jsx
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()
```

![image.png](image.png)

- 0에서 1까지 변하는 시그모이드 함수 확인

**로지스틱 회귀로 이진 분류**

- 이진 분류일 경우 시그모이드 함수의 출력이 0.5보다 크면 양성 클래스,

        0.5보다 작으면 음성 클래스로 판단

**불리언 인덱싱**: True, False 값을 전달하여 행을 선택

```jsx
 bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
```

- train_scaled와 train_target 배열에 불리언 인덱싱을 적용하여 도미와 빙어 데이터만 골라냄

```jsx
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))
```

모델 훈련 후 처음 5개 샘플을 예측: ['Bream' 'Smelt' 'Bream' 'Bream' 'Bream']

```jsx
print(lr.predict_proba(train_bream_smelt[:5]))
```

[[0.99760007 0.00239993]
[0.02737325 0.97262675]
[0.99486386 0.00513614]
[0.98585047 0.01414953]
[0.99767419 0.00232581]]

각각 음성 / 양성 클래스의 확률

```jsx
print(lr.classes_)
```

['Bream''Smelt'] : 빙어(Smelt)가 양성 클래스 

```jsx
print(lr.coef_, lr.intercept_)
```

z = -0,404 × (Weight) - 0.576 x (Length) - 0,663 × (Diagonal) - 1,013 x (Height) - 0,732 x (Width) - 2, 161

```jsx
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

from scipy.special import expit
print(expit(decisions))
```

처음 5개 샘플의 값을 출력한 뒤 시그모이드 함수에 통과시켜 확률을 얻음

[0.00240145  0.97264817  0.00513928  0.01415798  0.00232731]

**로지스틱 회귀로 다중 분류 수행**

 LogisticRegression 클래스를 사용해 7개의 생선을 분류

- 기본적으로 반복적인 알고리즘을 사용 → 충분하게 훈련시키기 위해 반복 횟수를 1,000으로 지정
- 릿지 회귀와 같이 계수의 제곱을 규제 → 완화를 위해 매개변수 C를 20으로 지정

```jsx
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```

0.9327731092436975          0.925

두 세트 모두 점수가 높고 과대적합 / 과소적합 발생하지 않음

```jsx
print(lr.predict(test_scaled[:5]))

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
```

 테스트 세트의 처음 5개 샘플에 대한 예측 확률을 출력

[[0.    0.014 0.842 0.    0.135 0.007 0.003]
[0.    0.003 0.044 0.    0.007 0.946 0.   ]
[0.    0.    0.034 0.934 0.015 0.016 0.   ]
[0.011 0.034 0.305 0.006 0.567 0.    0.076]
[0.    0.    0.904 0.002 0.089 0.002 0.001]]

7개 생선에 대한 확률을 계산했으므로 7개의 열이 출력

```jsx
print(lr.classes_)
```

['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']

- 첫 번째 샘플의 세 번째 열의 확률 84.1%로 가장 높음 → Perch
- 샘플마다 클래스 별로 확률 출력 → 가장 높은 확률이 예측 클래스가 됨

```jsx
print(lr.coef_.shape, lr.intercept_.shape)
```

- 5개의 특성을 사용하므로 coef 배열의 열은 5개
- intercept 7개 → z를 7개나 계산 ( 다중 분류는 클래스마다 값을 하나씩 계산)

**소프트맥수 함수**

다중 분류에서 z를 0과 1 사이의 확률값으로 변환

- e_sum = ez1 + ez2 + ez3 + ez4 + ez5 + ez6 + ez7
- 각각 e_sum으로 나누어 줌

```jsx
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))
```

테스트 세트의 처음 5개 샘플에 대한 z1~z7의 값

[[ -6.51   1.04   5.17  -2.76   3.34   0.35  -0.63]
[-10.88   1.94   4.78  -2.42   2.99   7.84  -4.25]
[ -4.34  -6.24   3.17   6.48   2.36   2.43  -3.87]
[ -0.69   0.45   2.64  -1.21   3.26  -5.7    1.26]
[ -6.4   -1.99   5.82  -0.13   3.5   -0.09  -0.7 ]]

```jsx
from scipy.special import softmax
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
```

소프트맥스 함수 계산 결과

[[0.    0.014 0.842 0.    0.135 0.007 0.003]
[0.    0.003 0.044 0.    0.007 0.946 0.   ]
[0.    0.    0.034 0.934 0.015 0.016 0.   ]
[0.011 0.034 0.305 0.006 0.567 0.    0.076]
[0.    0.    0.904 0.002 0.089 0.002 0.001]]

- 앞서 구한 proba 배열과 정확히 일치
- 로지스틱 회귀를 사용해 7개의 생선에 대한 확률을 예측하는 모델 훈련 성공

# 4-2. 확률적 경사 하강법

**점진적 학습**

- 한 번에 전체 데이터를 다 모아서 훈련할 수 없는 상황에서

       점점 도착하는 데이터를 반영해 예측 모델을 계속 개선하는 방법

## 확률적 경사 하강법

### 경사 하강법

- 손실 함수를 최소화하는 파라미터(모델)를 찾기 위한 방법
- 현재 위치에서 기울기(경사)를 계산하고 그 **반대 방향**으로 **조금씩 이동**
- 산에서 가장 가파른 내리막길을 따라 천천히 내려오는 것과 유사

| 종류 | 설명 | 장점 | 단점 |
| --- | --- | --- | --- |
| **배치 경사 하강법** (Batch GD) | 전체 훈련 데이터를 한 번에 사용해 한 번 이동 | 안정적, 정확한 방향 | 느리고 메모리 소모 큼 |
| **확률적 경사 하강법** (SGD) | 훈련 샘플 1개를 무작위로 골라 매번 이동 | 빠르고 메모리 효율적 | 진동 큼, 경로 불안정 |
| **미니배치 경사 하강법** (Mini-batch GD) | 여러 개의 샘플(소규모 배치)을 사용해 이동 | SGD보다 안정적, 실무에서 가장 많이 사용 | 하이퍼파라미터(배치 크기) 설정 필요 |
- 훈련 샘플 무작위 선택, 매번 하나의 샘플로 모델을 조금씩 업데이트
- 에포크: 모든 데이터를 한 번씩 사용하는 과정, 보통 수십~수백번 반복
- 한 번에 너무 크게 이동하면 최솟값을 지나칠 수 있어 학습률을 작게 설정해 천천히 수렴

### 손실 함수

- 머신러닝 알고리즘이 얼마나 엉터리인지를 측정하는 기준 → 작을수록 좋음
- 정확도: 맞았는가(1), 틀렸는가(0)로만 판단하는 이산적 값

        예측값을 조금 바꿔도 손실이 그대로일 수 있어 경사 하강법으로 조금씩 이동하는 것이 불가능

- 연속적인 손실함수가 필요함

### 로지스틱 손실 함수

| 타깃 yy | 예측 확률 y^\hat{y} | 사용되는 수식 | 손실의 해석 |
| --- | --- | --- | --- |
| 1 (양성) | 높을수록 좋음 (→1) | −log⁡(y^)-\log(\hat{y}) | 예측값이 1에 가까울수록 손실 작음 |
| 0 (음성) | 낮을수록 좋음 (→0) | −log⁡(1−y^)-\log(1 - \hat{y}) | 예측값이 0에 가까울수록 손실 작음 |

| 샘플 | 타깃 yy | 예측 확률 y^\hat{y} | 손실 함수 식 | 손실 값 대략 |
| --- | --- | --- | --- | --- |
| ① | 1 | 0.9 | −log⁡(0.9)-\log(0.9) | ≈ 0.105 |
| ② | 1 | 0.3 | −log⁡(0.3)-\log(0.3) | ≈ 1.204 |
| ③ | 0 | 0.2 | −log⁡(1−0.2)=−log⁡(0.8)-\log(1 - 0.2) = -\log(0.8) | ≈ 0.223 |
| ④ | 0 | 0.9 | −log⁡(1−0.9)=−log⁡(0.1)-\log(1 - 0.9) = -\log(0.1) | ≈ 2.302 |

정답에 가까울수록 손실은 작고, 틀릴수록 손실은 매우 커짐

→ 로지스틱 손실 함수 or 이진 크로스엔트로피 손실 함수

→ 다중 분류에서 사용되는 것은  크로스엔트로피 손실 함수

## SGDClassifier

데이터 준비

```jsx
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
```

```jsx
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
fish_input, fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

- Species 열을 제외한 나머지 5개는 입력 데이터로 사용, Species 열은 타깃 데이터
- 데이터를 훈련 세트와 테스트 세트로 나눔
- 훈련 세트와 테스트 세트의 특성을 표준화 전처리

```jsx
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```

- loss="log'로 지정하여 로지스틱 손실 함수를 지정
- 에포크 횟수를  10으로 지정하여 전체 훈련 세트를 10회 반복
- 0.773109243697479 , 0.775 → 낮은 정확도 → 10번이 부족한 것으로 보임

```jsx
sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```

- partial_fit () 메서드를 호출하고 다시 훈련 세트와 테스트 세트의 점수를 확인
- fit() 메서드와 사용법이 같지만 호출할 때마다 1 에포크씩 이어서 훈련할 수 있음
- 0.8151260504201681,  0.825  → 향상된 정확도

## 에포크와 과대/과소적합

- 에포크 횟수가 적으면 모델이 훈련 세트를 덜 학습
- 적은 에포크 횟수 동안에 훈련한 모델은 훈련 세트와 테스트 세트에

        잘 맞지 않는 과소적합된 모델일 가능성이 높음

- 반대로 많은 에포크 횟수 동안에 훈련한 모델은 훈련 세트에 너무

        잘 맞아 테스트 세트에는 과대적합된 모델일 가능성이 높음

- 훈련 세트 점수는 에포크가 진행될수록 꾸준히 증가하지만

       테스트 세트 점수는 어느 순간 감소하기 시작 → 과대적합 시작 지점

**조기 종료** : 과대적합이 시작하기 전에 훈련을 멈추는 것

```jsx
import numpy as np
sc = SGDClassifier(loss='log', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)

for i in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))
```

- 300번의 에포크 동안 훈련을 반복하여 진행

```jsx
import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```

![image.png](image%201.png)

- 백 번째 에포크 이후에는 훈련 세트와 테스트 세트의 점수가 조금씩 벌어짐
- 에포크 초기에는 과소적합되어 훈련 세트와 테스트 세트의 점수가 낮음
- 백 번째 에포크가 적절한 횟수로 보임

```jsx
sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```

0.957983193277311,  0.925

→ 반복 횟수를 100에 맞추고 모델을 다시 훈련한 결과 비고적 높은 점수가 나옴

**SGDClassifier의 loss 매개변수**

- 힌지 손실: 서포트 벡터 머신이라 불리는 또 다른 머신러닝 알고리즘을 위한 손실 함수
