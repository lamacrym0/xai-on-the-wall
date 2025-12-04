import ciu
import numpy as np
import numpy as np
import pandas as pd
import torch

def get_explainer_CIU(model, data,output_names, feature_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()   
    def torch_predict(X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_np = X.values.astype(np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)

        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        with torch.no_grad():
            X_tensor = torch.from_numpy(X_np).to(device)
            y_tensor = model(X_tensor)            

        y_np = y_tensor.detach().cpu().numpy()    
        return y_np
    X_train_df = pd.DataFrame(data["X_train"], columns=feature_names)

    dataset_df = X_train_df.copy()
    dataset_df["target"] = data["y_train"]

    CIU_model = ciu.CIU(
        predictor=torch_predict,
        out_names=output_names,
        data=X_train_df     
    )
    
    return CIU_model

def get_ciu_instance(CIU_model, instance):
    res = CIU_model.explain(
        instance=instance,
    )
    return res