from test.cases import forward_propagation_with_dropout_test_case
from app.DNN import DNN

dnn_obj = DNN(keep_probs=[1, 0.7, 0.7], layer_dims=[2, 3, 1])
X_assess, parameters = forward_propagation_with_dropout_test_case()

A3, cache = dnn_obj.dnn_forward_prop(X_assess, parameters)
print("A3 = " + str(A3))
