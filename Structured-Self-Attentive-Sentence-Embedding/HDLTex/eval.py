from keras import metrics
import numpy as np
from sklearn.metrics import accuracy_score

def evaluate(l1_model,l2_models,L2_dict_i2y,x_test,y_test):
    """this model evaluate the trained hierarchical models"""
    l1_y_pred,l2_y_pred = [],[]
    l1_y_true,l2_y_true = [],[]
    for i in range(len(y_test)):
        x_t = np.array([x_test[i]])
        l1_true,l2_true = y_test[i][0,0], y_test[i][0,1]
        
        l1_pred = np.argmax(l1_model.predict(x_t)[0]) # get the pred from l1 model
        model2 = l2_models[l1_pred] #find the right l2 model
        l2_i = np.argmax(model2.predict(x_t)[0])
        l2_pred = L2_dict_i2y[l1_pred][l2_i]
        
        #append
        l1_y_pred.append(l1_pred)
        l1_y_true.append(l1_true)
        l2_y_pred.append(l2_pred)
        l2_y_true.append(l2_true)
    
    l1_acc = accuracy_score(l1_y_true, l1_y_pred)
    l2_acc = accuracy_score(l2_y_true, l2_y_pred)
    print("l1 acc: %.4f,  l2 acc: %.4f "%(l1_acc,l2_acc))
    return l1_acc, l2_acc
    # return metrics.categorical_accuracy(y_test, y_preds)