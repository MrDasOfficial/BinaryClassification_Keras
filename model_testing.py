import numpy as np
from keras.models import load_model
from model_training import x_test,y_test


loaded_model = load_model("model_diabetics.h5")

prediction = loaded_model.predict(x_test)
print(prediction)
# the predicted values might be between 0 and 1
# value above 0.5 is considered as 1 ; diabetic
# value below 0.5 is considered as 0 ; non-diabetic

for result in prediction:
    if result>0.5:
        print(f"{result} the patient appears to be diabetic")
    elif result<0.5:
        print(f"{result} the patient appears to be non-diabetic")

# to compare the result with real output y_test
# this shows the accuracy of the tested model
for model_pred,actual_pred in zip(prediction,y_test):
    if model_pred>0.5:
        print(f" 1 ==> {actual_pred}")
    elif model_pred<0.5:
        print(f" 0 ==> {actual_pred}")
