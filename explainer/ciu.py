import ciu
import numpy as np

def get_explain_CIU(model, data,out_name):
    pred_test_prices = model.predict(data["X_test"])
    inst_ind = np.argmax(pred_test_prices) # Test set instance with highest price estimate, so we expect a "positive" explanation.
    instance = data["X_test"].iloc[[inst_ind]]
    CIU = ciu.CIU(model.predict, out_name, data=data["X_train"])
    return CIU.explain(instance)