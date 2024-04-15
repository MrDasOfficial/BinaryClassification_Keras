
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

data = np.loadtxt("pima-indians-diabetes.csv",delimiter=",") # loading dataset, delimiter is used for avoiding ','

x_input = data[:,:-1]  # slicing ; all the rows - 0 to 7th column (avoiding last column)
y_output = data[:,-1]  # slicing ; all the rows - last column only
print(x_input.shape)
# shape of x_input --> (768,8)
# shape of y_output --> (8,)

# need to split the data into train or test data
#  760 rows for training
x_train = x_input[:760]
y_train = y_output[:760]

# remaining 8 rows for testing
x_test = x_input[760:]
y_test = y_output[760:]

# creating model by adding layers ------------

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






