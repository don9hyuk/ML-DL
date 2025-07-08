# Ch.3 회귀 알고리즘과 모델 규제

회귀 알고리즘과 모델 규제

# 3-1. k-최근접 이웃 회귀

한빛 마켓은 여름 농어철을 맞아 농어를 **무게 단위로 판매**하기로 한다.

기존에는 마리당 가격으로 판매했지만, 고객들이 기대보다 작은 농어를 받았다고 항의하는 일이 생겨 무게 기준이 더 합리적이라고 판단한 것이다.

그런데 공급처에서 **농어 무게를 잘못 측정**해보내는 바람에 문제가 생긴다.

다행히 길이, 높이, 두께 같은 다른 데이터는 정확하게 측정되어 있어 이를 활용해 **무게를 예측하는 모델**을 만들어보기로 한다.

## k-최근접 이웃 회귀 (k-NN Regression)

- **k-NN 분류**는 주변에 가까운 샘플 k개를 찾아 **가장 많은 클래스**를 정답으로 선택
- **k-NN 회귀**는 주변의 샘플 k개의 수치(target 값)를 평균 내서 예측값으로 사용

ex) 가까운 이웃 3개의 무게가 100g, 80g, 60g이라면, 예측값은 평균인 **80g**

```jsx
##데이터준비
import numpy as np
perch_length = np.array(
[ 8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
40.0, 42.0, 43.0, 43.0, 43.5, 44.0])
perch_weight = np.array(
[5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
1000.0, 1000.0]
)
```

- 길이와 무게의 산점도

```jsx
import matplotlib.pyplot as plt
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

![image.png](image.png)

-당연하게도 농어의 길이가 늘어날수록 무게도 늘어남

```jsx
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
perch_length, perch_weight, random_state=42
```

-사이킷런의 train_test_split () 함수를 사용해 훈련 세트와 테스트 세트로 나눔

```jsx
test_array = np.array([1,2,3,4])
print(test_array.shape)
test_array = test_array.reshape(2, 2)
print(test_array.shape)
```

```jsx
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
print(train_input.shape, test_input.shape)
```

- 사이킷런에 사용할 훈련 세트는 2차원 배열이어야 함 → 수동으로 변경
- reshape (-1, 1)과 같이 사용하면 배열의 전체 원소 개수를 매번 외우지 않아도 되므로 편리함

### 모델 훈련 및 결정계수 R²

- KNeighbors Regressor: 사이킷런에서 k-최근접 이웃 회귀 알고리즘을 구현한 클래스

```jsx
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
#k-최근접 이웃 회귀 모델을 훈련합니다
knr.fit(train_input, train_target)
```

```jsx
 print(knr.score(test_input, test_target))
```

0.9928094061010639의 점수값: **결정계수 R²**

- 모델이 타깃값의 변동성을 얼마나 잘 설명하는지 나타냄
- 1에 가까울수록 예측이 잘 된것

### 평균 절댓값 오차 (MAE, Mean Absolute Error)

- 각 샘플에서 예측값과 실제값의 차이(절댓값)를 모두 더한 뒤, 그 평균을 구한 값

```jsx
from sklearn.metrics import mean_absolute_error
#테스트 세트에 대한 예측을 만듭니다.
test_prediction = knr.predict(test_input)
#테스트 세트에 대한 평균 절댓값 오차를 계산합니다
mae = mean_absolute_error(test_target, test_prediction)
print(mae)
```

19.157142857142862: 예측이 평균적으로 19g 정도 타깃값과 다르다는 것

### 과대적합 vs 과소적합

```jsx
print(knr.score(train_input, train_target))
```

훈련 세트보다 테스트 세트의 점수가 높으니 **과소적합**

- **과소적합**

       훈련 세트보다 테스트 세트의 점수가 높거나 두 점수가 모두 너무 낮은 경우

      모델이 너무 단순하여 훈련 세트에 적절히 훈련되지 않은 경우

- **과대적합**

       훈련 세트에서 점수가 굉장히 좋았는데 테스트 세트에서는 점수가 굉장히 나쁜 경우

       훈련 세트에만 잘 맞는 모델이라 테스트 세트와 나중에 실전에 투입하여 새로운 샘플에 대한 예측을 

       만들기 부적합한 경우

### 과소적합 해결

모델을 조금 더 복잡하게 만들어 과소적합 해결

- k-최근접 이웃 알고리즘으로 모델을 더 복잡하게 만드는 방법은 이웃의 개수 k를 줄이는 것

```jsx
#이웃의 개수를 3으로 설정합니다.
knr.n_neighbors = 3
#모델을 다시 훈련합니다.
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
```

k 값을 줄였더니 훈련 세트의 **R²** 점수가 높아짐

```jsx
print(knr.score(test_input, test_target))
```

- 테스트 세트의 점수는 훈련 세트보다 낮아졌으므로 과소적합 문제를 해결함
- 두 점수의 차이가 크지 않으므로 이 모델이 과대적합도 아님

# 3-2. 선형 회귀

## k-최근접 이웃 회귀의 한계

훈련 데이터 범위 밖의 예측에서 문제가 생김

- `perch_length` 데이터를 보면 가장 긴 농어가 44.0cm에 불과
- 모델이 본 적 없는 구간에서 예측을 하게 된 것
- 가장 가까운 샘플도 40cm 근처에 있어 최근접 방법에 한계가 생김

```jsx
print(knr.predict([[50]]))
```

길이가 50cm인 농어의 무게를 예측했을 때 실제보다 훨씬 작은 값이 나옴

```jsx
import matplotlib.pyplot as plt
#50cm 농어의 이웃을 구합니다
distances, indexes = knr.kneighbors([[50]])
#훈련 세트의 산점도를 그립니다.
plt.scatter(train_input, train_target)
#훈련 세트 중에서 이웃 샘플만 다시 그립니다.
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
#50cm 농어 데이터
plt.scatter(50, 1033, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

![image.png](image%201.png)

- 훈련 세트와 50cm 농어 그리고 이 농어의 최근접 이웃의 산점도
- k-최근접 이웃 회귀는 가장 가까운 샘플을 찾아 타깃을 평균
- 따라서 새로운 샘플이 훈련 세트의 범위를 벗어나면 엉뚱한 값을 예측할 수 있음
예를 들어 길이가 100cm인 농어도 여전히 1,033g으로 예측
- → 다른 알고리즘 필요!

## 선형 회귀

- 특성이 하나인 경우 어떤 직선을 학습하는 알고리즘
- 비교적 간단하고 성능이 뛰어남

```jsx
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#선형 회귀 모델을 훈련합니다
lr.fit(train_input, train_target)
#50cm 농어에 대해 예측합니다
print(lr.predict([[50]]))
```

k-최근접 이웃 회귀를 사용했을 때와 달리 선형 회귀는 50cm 농어의 무게를 아주 높게 예측

![KakaoTalk_20250708_094145847.png](KakaoTalk_20250708_094145847.png)

x를 농어의 길이, 를 농어의 무게로 선형 회귀가 학습한 직선

```jsx
 print(lr.coef, lr.intercept_)
```

기울기: 39.01714496         절편: -709.0186449535477

```jsx
#훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)
#15에서 50까지 1차 방정식 그래프를 그립니다
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])
# 50cm 농어 데이터
plt.scatter(50, 1241.8, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

![image.png](image%202.png)

선형 회귀 알고리즘이 이 데이터셋에서 찾은 최적의 직선 → 50cm의 농어 무게도 예측 가능

```jsx
print(lr.score(train_input, train target)) # 훈련 세트
print(lr.score(test_input, test_target)) #테스트 세트
```

- 훈련 세트와 테스트 세트의 점수 차이 → 전체적으로 과소적합
- 이 직선대로 예측하면 농어의 무게가 0g 이하로 나가는 불가능한 일 발생

## 다항 회귀

- 최적의 직선을 찾기보다 최적의 곡선을 찾기로 변경
- 2차 방정식의 그래프를 그리기 위해 이를 제곱한 항을 훈련 세트에 추가

```jsx
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
print(train_poly.shape, test_poly.shape)
```

- 길이를 제곱하여 왼쪽 열에 추가했기 때문에 훈련 세트와 테스트 세트 모두 열이 2개로 변경

- train_poly를 사용해 선형 회귀 모델을 다시 훈련
- 2차 방정식 그래프를 찾기 위해 훈련 세트에 제곱 항을 추가했지만, 타깃값은 그대로 사용

```jsx
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))
```

[1573.98423528] → 이전 모델보다 더 높은 예측값

```jsx
print (lr.coef_, lr.intercept_)
```

- 무게 = 1.01 × 길이 - 21.6 × 길이 + 116.05
- 이 다항식을 사용하여 선형 회귀 → 다항 회귀

```jsx
#구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듭니다.
point = np.arange(15, 50)
#훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)
#15에서 49까지 2차 방정식 그래프를 그립니다
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
#50cm 농어 데이터
plt.scatter(50, 1574, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

![image.png](image%203.png)

- 산점도를 그린 결과 훈련 세트의 경향을 잘 따르고 있고 무게도 정상적으로 나옴

```jsx
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
```

 R² 점수를 평가: 훈련 세트와 테스트 세트에 대한 점수가 크게 높아짐

# 3-3. 특성 공학과 규제

지금까지는 길이와 길이²만 사용한 2차 다항 회귀를 썼음.

→ 그런데 여전히 과소적합이 남아있음

→ 알고 보니 **높이와 두께 데이터도 있었음** → 특성이 부족했던 것

## 다중 회귀

- 여러 개의 특성을 사용한 선형 회귀
- **특성 수와 선형 회귀 모델의 형태**

| 특성 수 | 모델이 학습하는 형태 | 수학적 표현 |
| --- | --- | --- |
| 1개 | 직선 (line) | 𝑦 = 𝑎𝑥 + 𝑏 |
| 2개 | 평면 (plane) | 𝑦 = 𝑎₁𝑥₁ + 𝑎₂𝑥₂ + 𝑏 |
| 3개 이상 | 초평면 (hyperplane) | 𝑦 = 𝑎₁𝑥₁ + 𝑎₂𝑥₂ + 𝑎₃𝑥₃ + ... + 𝑏 |
- 3차원 이상은 시각화할 수 없지만, 수학적으로는 완전히 처리할 수 있음
- 선형 회귀는 특성이 많아질수록 훨씬 더 복잡한 경계를 표현할 수 있는 모델

**특성 공학**

- 단순한 `길이`, `높이`, `두께` 3개만 쓰는 게 아니라

        그 제곱값, 곱셈 조합(상호작용 항) 등을 만들어 특성을 확장

- 특성을 확장하면, 선형 회귀로도 비선형적인 복잡한 관계를 표현 가능
- 고차항을 추가해도 여전히 선형 회귀는 학습과 해석이 빠르고 안정적

```jsx
import pandas as pd 
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)

import numpy as np
perch_weight = np.array([
5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
1000.0, 1000.0])

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
perch_full, perch_weight, random_state=42)
```

- 데이터 준비 후 perch_full과 perch_weight를 훈련 세트와 테스트 세트로 나눔

**변환기**: 사이킷런에서  특성을 만들거나 전처리하기 위한 다양한 클래스를 제공

```jsx
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures()
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))
```

2개의 특성(원소)을 가진 샘플 [2, 3]이 6개의 특성을 가진 샘플 [1. 2.3.4.6.9.]로 변환됨

- 무게 = ax 길이 + bx 높이 +cx 두께 + dx 1
- 각 특성을 제곱한 항을 추가하고 특성끼리 서로 곱한 항을 추가

```jsx
poly = PolynomialFeatures(include_bias=False)
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))
```

절편을 위한 항이 제거되고 특성의 제곱과 특성끼리 곱한 항만 추가 (1 제거)

```jsx
poly = PolynomialFeatures (include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)
```

```jsx
poly.get_feature_names_out()
```

9개의 특성 만들어진 것 확인 → 9개의 특성 조합 확인

['x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2', 'x2^2']

```jsx
test_poly = poly.transform(test_input)
```

테스트 세트 변환 완료

## 다중 회귀 모델 훈련

```jsx
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.score(train_poly, train_target))
```

0.9903183436982124 : 아주 높은 점수 → 특성이 늘어날수록 선형 회귀의 능력도 강화됨

```jsx
print(lr.score(test_poly, test_target))
```

테스트 점수는 높아지지는 않았지만 과소적합 문제는 발생하지 않음

```jsx
poly = PolynomialFeatures (degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)

lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
```

특성을 55개로 늘려 훈련 결과 0.9999…의 아주 높은 값이 나옴

```jsx
print(lr.score(test_poly, test_target))
```

그러나 테스트 점수는 -144.40579436844948로 아주 큰 음수가 됨

→ 특성의 개수를 크게 늘리면 선형 모델은 훈련 세트에 대해 거의 완벽하게 학습할 수 있지만 

    이런 모델은 훈련 세트에 너무 과대적합 되므로 테스트 세트에서는 효과가 없음

## 규제

- 모델이 훈련 세트에 과대적합되지 않도록 만드는 것
- 선형 회귀 모델의 경우 특성에 곱해지는 계수(또는 기울기)의 크기를 작게 만드는 일

55개의 특성으로 훈련한 선형 회귀 모델의 계수를 규제

→ 훈련 세트의 점수를 낮추고 대신 테스트 세트의 점수를 높이자

- 일반적으로 선형 회귀 모델에 규제를 적용할 때 계수 값의 크기가 많이 다르면 공정하게 제어되지 않음
- 규제를 적용하기 전에 먼저 정규화 필요

```jsx
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
```

### 릿지 회귀

```jsx
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
```

0.97906939… → 이전 점수보다 약간 낮아진 점수

```jsx
 print(ridge.score(test_scaled, test_target))
```

0.9790693977615398 → 정상으로 돌아온 테스트 점수

많은 특성을 사용했음에도 불구하고 훈련 세트에 너무 과대적합되지 않아 

테스트 세트에서도 좋은 성능을 보임

- alpha 매개변수로 규제의 강도를 조절
- alpha 값이 크면 규제 강도가 세지므로 계수 값을 더 줄이고 조금 더 과소적합되도록 유도
- alpha 값이 작으면 계수 감소가 줄고 선형 회귀 모델과 유사해지므로 과대적합될 가능성이 큼

```jsx
import matplotlib.pyplot as plt
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    #릿지 모델을 만듭니다
    ridge = Ridge(alpha=alpha)
    #릿지 모델을 훈련합니다.
    ridge.fit(train_scaled, train_target)
    #훈련 점수와 테스트 점수를 저장합니다
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))
```

- alpha 값을 0.001에서 100까지 10배씩 늘려가며 릿지 회귀 모델을 훈련한 다음

       훈련 세트와 테스트 세트의 점수를 파이썬 리스트에 저장

```jsx
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```

![image.png](image%204.png)

- 위: 훈련 세트 그래프 / 아래: 테스트 세트 그래프
- 그래프의 왼쪽: 훈련 세트에는 잘 맞고 테스트 세트에는 과대적합의 모습
- 그래프의 오른쪽: 훈련 세트와 테스트 세트의 점수가 모두 낮아지는 과소적합

```jsx
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
```

alpha 값 0.1로 훈련 결과 점수가 모두 높고 과대적합 / 과소적합이 발생하지 않음

### 라쏘 회귀

```jsx
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))

print(lasso.score(test_scaled, test_target))
```

훈련 점수, 테스트 점수 둘 다 좋음

```jsx
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
#라쏘 모델을 만듭니다.
lasso = Lasso(alpha=alpha, max_iter=10000)
#라쏘 모델을 훈련합니다.
lasso.fit(train_scaled, train_target)
# 훈련 점수와 테스트 점수를 저장합니다.
train_score.append(lasso.score(train_scaled, train_target))
test_score.append(lasso.score(test_scaled, test_target))

```

alpha 값을 바꾸어 가며 점수 계산

```jsx
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```

![image.png](image%205.png)

왼쪽은 과대적합, 오른쪽으로 갈수록 훈련 세트와 테스트 세트의 점수가 좁혀지고 있음

라쏘 모델에서 최적의 alpha 값은 1, 즉 10*1=10

```jsx
 lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
```

라쏘 모델이 과대적합을 잘 억제하고 테스트 세트의 성능을 크게 높인 결과

```jsx
print(np.sum(lasso.coef_ == 0))
```

40개의 값이 0이 됨 → 55개의 특성을 모델에 주입했지만 라쏘 모델이 사용한 특성은 15개

→ 라쏘 모델을 유용한 특성을 골라내는 용도로도 사용 가능
