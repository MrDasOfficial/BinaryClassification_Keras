
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

data = np.loadtxt("pima-indians-diabetes.csv",delimiter=",") # loading dataset, delimiter is used for avoiding ','

x_input = data[:,:-1]
y_output = data[:,-1]


# print(x_input)
# print(y_output)

# model ------------

model = Sequential()
model.add(Dense(12,activation = "relu", input_shape=(8,)))
model.add(Dense(8,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

# relu rectified linear unit activation ==== > f(x) = max(0,x)
# f(2) = max(0,2) ==> 2, f(-2) = max(0,-2) ==> 0

# sigmoid ====> s(x) = 0<ans>1.... 1/(1+e^-x)
#  e = 2.714 Euler's constant
# s(2) ==>  0.88

model.compile(loss="binary_crossentropy",optimizer="adam",metrics = ["accuracy"])
model.fit(x_input,y_output,epochs = 100,batch_size=10)






