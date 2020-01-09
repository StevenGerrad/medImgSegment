
########################################################################################
# 
#   Date: 2019.11.9
#   --用了一上午jupyter感觉不是很趁手，换回VScode
#   使用 keras 尝试训练fashion-mnist
#       https://www.jianshu.com/p/e9c1e68a615e  
# 
########################################################################################

import keras
from keras.models import Sequential
from keras.layers import Dense
import mnistRead

num_classes = 10
x_train,y_train,x_test,y_test = mnistRead.get_data()

X_train = x_train.reshape(x_train.shape[0], -1) / 255. 
X_test = x_test.reshape(x_test.shape[0], -1) / 255.  
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(units=256, activation='relu', input_dim=784))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
# 进一步地配置优化器
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

# x_train 和 y_train 是 Numpy 数组 -- 就像在 Scikit-Learn API 中一样。
print('Training ------------\n')
model.fit(x_train, y_train, epochs=5, batch_size=50)
# 或者，你可以手动地将批次的数据提供给模型：
# model.train_on_batch(x_batch, y_batch)

# 只需一行代码就能评估模型性能：
print('Testing ------------\n')
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

print('test: ', loss_and_metrics)