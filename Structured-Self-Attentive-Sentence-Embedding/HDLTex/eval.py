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


def evaluateDB(l1_model,l2_models,l3_models,d_train,df_test):
    """this model evaluate the trained hierarchical models"""
    l1_y_pred, l2_y_pred, l3_y_pred = [], [], []
    for i in range(len(df_test)):
        x_t = np.array(df_test.text[i])
        print(x_t)
        l1_true, l2_true, l3_true = df_test.l1[i],df_test.l2[i],df_test.l3[i]

        l1_pred_i = np.argmax(l1_model.predict(x_t)[0])  # get the pred from l1 model
        model2 = l2_models[l1_pred_i]  # find the right l2 model
        l2_pred_i = np.argmax(model2.predict(x_t)[0])
        model3 = l3_models[l1_pred_i][l2_pred_i]
        l3_pred_i = np.argmax(model3.predict(x_t)[0])

        #reainslate to the ture labels
        l1_pred = d_train.dict.i2y[l1_pred_i]
        l2_pred = d_train.childs[l1_pred_i].dict.i2y[l2_pred_i]
        l3_pred = d_train.childs[l1_pred_i].childs[l2_pred_i].dict.i2y[l3_pred_i]

        # append
        l1_y_pred.append(l1_pred)
        l2_y_pred.append(l2_pred)
        l3_y_pred.append(l3_pred)

    l1_acc = accuracy_score(df_test.l1, l1_y_pred)
    l2_acc = accuracy_score(df_test.l2, l2_y_pred)
    l3_acc = accuracy_score(df_test.l3, l3_y_pred)
    # print("l1 acc: %.4f,  l2 acc: %.4f , l3 acc: %.4f" % (l1_acc, l2_acc, l3_acc))
    return l1_acc, l2_acc, l3_acc
    # return metrics.categorical_accuracy(y_test, y_preds)