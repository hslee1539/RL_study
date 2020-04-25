# tf.keras.Model.`__call__`

```python
model = Model()
model() # = Model.__call__()
```

## 요약
### 인수
| 타입 | 이름 | 내용 |
| -- | -- | -- |
| `Tuple[tf.Variable]` | `inputs` | 모델을 이 값으로 forward합니다. 튜플 안에는 `np.ndarray`이나 `tf.Tensor`를 사용 못합니다. |
### 반환
모델의 output값으로 `tf.Tensor` 객체를 반환합니다.

```python
import tensorflow as tf
import numpy as np

# xor
x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

x1 = x[:,0].reshape((-1,1))
x2 = x[:,1].reshape((-1,1))

input_layer1 = tf.keras.layers.Input(1)
input_layer2 = tf.keras.layers.Input(1)
concat_layer = tf.keras.layers.concatenate([input_layer1, input_layer2])
out_layer = tf.keras.layers.Dense(1)

model = Model(inputs=[input_layer1, input_layer2], outputs=out_layer)
model([x1, x2]) # 에러
model([tf.Variable(x1), tf.Variable(x2)]) # ok
```

## `np.ndarray`를 못받는 이유
추측하기로는 np.ndarray는 텐서플로우의 계산 노드에 있을 것 같습니다.
