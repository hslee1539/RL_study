import tensorflow as tf
import numpy as np

from typing import Tuple

def Sequential(input : tf.Tensor, *layers : tf.keras.layers.Layer) -> Tuple[tf.Tensor]:
    """
    레이어들을 연결을 합니다.

    ## 인수

    | 타입 | 이름 | 내용 |
    | -- | -- | -- |
    | `tf.Tensor`               | `input`   | 입력 텐서입니다. |
    | `tf.keras.layers.Layer`   | `layers`  | 연속적으로 연결할 레이어 객체들입니다. |

    ## 반환

    인수로 받은 `input` 마지막 레이어의 `tf.Tensor`를 반환합니다.
    """
    
    if len(layers) > 0:
        output = input
        for layer in layers:
            output = layer(output)
        return input, output
    return None, None

if __name__ == "__main__":
    ## 테스트 용
    layer_in1 = tf.keras.layers.Input(1)
    layer_in2 = tf.keras.layers.Input(1)
    
    layer_in, layer_out = Sequential(
        tf.keras.layers.concatenate([layer_in1, layer_in2]),
        #tf.keras.layers.Input(2),
        tf.keras.layers.Dense(5, activation="relu", kernel_initializer="he_uniform"),
        #tf.keras.layers.Flatten(),
        tf.keras.layers.GaussianNoise(1.0),
        tf.keras.layers.Dense(1, activation="relu", kernel_initializer="he_uniform")
    )
    model = tf.keras.Model(inputs=[layer_in1, layer_in2], outputs=layer_out)
    x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    x1 = x[:,0].reshape((-1,1))
    x2 = x[:,1].reshape((-1,1))
    x1_tensor = tf.Variable(x1)
    x2_tensor = tf.Variable(x2)
    print(model([x1_tensor, x2_tensor]))
    y = np.array([[0],[1],[1],[0]], dtype=float)
    print(model.predict([x1,x2]))
    #print(model(x.reshape(-1)))
    model.summary()
    opt = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError()
    print(len(model.layers))
    #model.compile(optimizer="adam", loss="mse")
    with tf.GradientTape() as tape:
        #print(model.layers[0](x1))
        #tape.watch(x1_tensor)
        #i1 = model.layers[0](x1)
        #i2 = model.layers[1](x2)
        #l1 = model.layers[2]([i1, i2])
        #print(l1)
        #l2 = model.layers[3](l1)
        #print(l2)
        #l3 = model.layers[4](l2)
        #out = model.layers[5](l3)
        
        out = model([x1_tensor, x2_tensor])
        #print(out)
        loss_Value = loss(y, out) + 1
    #print(loss_Value)
    grad = tape.gradient(loss_Value, x1_tensor)
    print(grad)
    #print(tape.watched_variables())
    
    #print(model.layers[2]([i1, i2]))
    #y_pred = 
    
    #loss_value = loss(y, y_pred)
    #grad = tape.gradient(loss_value, model.trainable_variables)
    #print(grad)
    #tape.watch(model.variables)
    #print(tape.watched_variables())
    #model.fit([x1,x2],y, epochs=100, verbose=0) # 에러
    #print(tape.watched_variables())
        
    model.fit([x1,x2], y)
    print(model.predict([x1,x2]))
    #grad = tf.keras.backend.function([layer_in1, layer_in2], tf.keras.backend.gradients(layer_out, [layer_in1]))
    
    #print(model.predict([x1,x2]))

