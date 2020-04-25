# class DQN

## 요약
DQN 클래스입니다.


------------------------------------------------------------------
## 필드
### __num_inputs
```python
class DQN:
    def __init__(self, num_inputs : int, num_outputs : int):
        ... ...
        self.__num_inputs = num_inputs
        ... ...
```
#### 요약
입력 노드의 수 값입니다.

### __num_outputs
```python
class DQN:
    def __init__(self, num_inputs : int, num_outputs : int):
        ... ...
        self.__num_outputs = num_outputs
        ... ...
```
#### 요약
출력 노드의 수 값입니다.

### __model
```python
class DQN:
    def __init__(self, num_inputs : int, num_outputs : int):
        ... ...
        self.__model = DQN.createModel(num_outputs)
        ... ...

```
#### 요약
`tensorflow.keras.Sequential` 객체로, DQN에서 모델로 사용합니다.

### __target_model
```python
class DQN:
    def __init__(self, num_outputs : int):
        ... ...
        self.__target_model = DQN.createModel(num_outputs)
        ... ...
```
#### 요약
`tensorflow.keras.Sequential` 객체로, DQN에서 타겟으로 사용합니다.

### __buffer
```python
class DQN:
    def __init__(self, num_outputs : int):
        ... ...
        self.__buffer = (
            np.ndarray()
        )
        ... ...
```
#### 요약
`deque` 객체로, DQN에서 학습할 데이터로 사용됩니다.

### __epsilon
```python
class DQN:
    def __init__(selfm num_outputs : int):
        ... ...
        self.__epsilon = 1.0
        ... ...
```
#### 요약
이 비율 만큼 랜덤적으로 이동합니다. e-greedy 기법을 참고하세오. [0,1]값을 설정합니다.

### __epsilon_decap
```python
class DQN:
    def __init__(self, num_outputs : int):
        ... ...
        self.__epsilon_decap = 0.99
        ... ...
```
#### 요약
액션동작을 할 수록 `__epsilon`의 값을 점진적으로 낮게 하기 위한 값입니다. [0,10)의 값을 설정합니다.

### __epsilon_min
```python
class DQN:
    def __init__(self, num_outputs : int):
        ... ...
        self.__epsilon_min = 0.001
        ... ...
```

#### 요약
`__epsilon`의 최소 값입니다.



--------------------------------------------------------------------
## 메소드
### act(self, state : np.ndarray)
```python
class DQN:
    ... ...
    def act(self, state : np.ndarray):
        ... ...
```
#### 요약
액션 동작을 계산, 처리, 반환합니다.

#### 인수
**self** : DQN의 객체입니다.

**state** : `np.ndarray`를 받고, 환경의 상태를 받습니다.

#### 반환
argmax값을 반환합니다.

### createModel(num_outputs : int)
```python
class DQN:
    @staticmethod
    def createModel(num_outputs : int):
        ... ...
```
#### 요약
모델을 만듭니다.

#### 인수
<b>num_outputs</b> : 정수 값을 받습니다.

#### 반환
내부에서 사용할 `tensorflow.keras.Sequential` 객체를 반환합니다.

#### 예제
```python
model = DQN.createModel(4)
```

### remember(self, state, action, reward, new_state, done)