import pandas as pd
import gradio as gr
import torch
import json
import os
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

from src.load_data import load_data
from src.model_builder import build_mlp_from_layer_df
from src.train import prepare_data

# ─────────────────────────────────────────────────────────────────────────────
# FILE & SAVES MANAGER
# ─────────────────────────────────────────────────────────────────────────────

SAVES_DIR = "saves"
MODELS_DIR = os.path.join(SAVES_DIR, "models")
IMAGE_DIR = os.path.join(SAVES_DIR, "images")
DATA_DIR = os.path.join(SAVES_DIR, "data")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def list_saved_runs():
    if not os.path.exists(SAVES_DIR):
        return []
    files = [f for f in os.listdir(SAVES_DIR) if f.endswith(".json")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(SAVES_DIR, x)), reverse=True)
    return files

def save_run_logic(run_name, data_state, layer_config, epochs, lr, seed, model, history, dexire_res, optim_name, loss_name,img_struct):
    """
    Saves the current run configuration, model, and results to disk.
    Now includes optimizer, loss function, and output dimension.
    """
    if not model or not data_state:
        return "Error: No model or data to save.", gr.update()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{run_name.replace(' ', '_')}_{timestamp}"
    
    # 1. DATASET
    ds_config = data_state["config"].copy()
    if ds_config["source"] == "CSV File" and "path" in ds_config:
        original_path = ds_config["path"]
        if os.path.exists(original_path):
            new_csv_name = f"{safe_name}_data.csv"
            new_csv_path = os.path.join(DATA_DIR, new_csv_name)
            shutil.copy(original_path, new_csv_path)
            ds_config["saved_path"] = new_csv_path
            ds_config["original_filename"] = os.path.basename(original_path)

    # 2. MODEL WEIGHTS
    model_filename = f"{safe_name}.pth"
    model_path = os.path.join(MODELS_DIR, model_filename)
    torch.save(model.state_dict(), model_path)

    # 3. IMAGE STRUCTURE
    image_filename = f"{safe_name}_structure.png"
    image_path = os.path.join(IMAGE_DIR, image_filename)
    img_struct = Image.open(img_struct)
    img_struct.save(image_path)
    
    
    # 4. METADATA JSON
    layers_data = layer_config.values.tolist() if hasattr(layer_config, "values") else layer_config
    
    # Calculate output dimension based on targets
    y_data = data_state.get("y", [])
    if len(y_data) > 0:
        n_classes = len(set(y_data))
        out_dim = n_classes if n_classes > 2 else 1
    else:
        out_dim = 1 

    session_data = {
        "meta": {
            "run_name": run_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_file": model_filename,
            "image_file": image_filename,
        },
        "dataset": ds_config,
        "architecture": {
            "type": "MLP",
            "input_size": getattr(model, "input_size", 0),
            "output_size": out_dim,  
            "layers": layers_data
        },
        "training": {
            "epochs": epochs,
            "lr": lr,
            "seed": int(seed) if seed is not None else 0,
            "optimizer": optim_name,     
            "loss_function": loss_name   
        },
        "history": history,
        "results": {
            "dexire": dexire_res
        }
    }
    
    json_path = os.path.join(SAVES_DIR, f"{safe_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=4)
        
    return f"Saved: {safe_name}", gr.update(choices=list_saved_runs()), gr.update(visible=False)

def overwrite_run_logic(json_filename, data_state, layer_config, epochs, lr, seed, model, history, dexire_res, optim_name, loss_name, img_struct):
    """
    Overwrites an existing run (JSON, Model, Image) with the current state.
    """
    if not json_filename:
        return "Error: No run selected to overwrite.", gr.update(), gr.update()
    
    if not model or not data_state:
        return "Error: No model or data to save.", gr.update(), gr.update()

    json_path = os.path.join(SAVES_DIR, json_filename)
    if not os.path.exists(json_path):
        return f"Error: File {json_filename} not found.", gr.update(), gr.update()

    # 1. LOAD EXISTING METADATA 
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        
        meta = old_data.get("meta", {})
        model_filename = meta.get("model_file")
        image_filename = meta.get("image_file")
        original_timestamp = meta.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        run_name = meta.get("run_name", "Unknown")
        
        if not model_filename or not image_filename:
            return "Error: Corrupted save metadata (missing filenames).", gr.update()
            
    except Exception as e:
        return f"Error reading existing file: {e}", gr.update()

    # 2. OVERWRITE MODEL WEIGHTS
    model_path = os.path.join(MODELS_DIR, model_filename)
    torch.save(model.state_dict(), model_path)

    # 3. OVERWRITE IMAGE STRUCTURE
    image_path = os.path.join(IMAGE_DIR, image_filename)
    try:
        if isinstance(img_struct, str):
            img_obj = Image.open(img_struct)
        else:
            img_obj = img_struct
        img_obj.save(image_path)
    except Exception as e:
        return
    # 4. PREPARE DATASET CONFIG
    ds_config = data_state["config"].copy()
    if ds_config["source"] == "CSV File" and "path" in ds_config:
        original_path = ds_config["path"]
        if os.path.exists(original_path):
            ds_config["original_filename"] = os.path.basename(original_path)

    # 5. UPDATE METADATA JSON
    layers_data = layer_config.values.tolist() if hasattr(layer_config, "values") else layer_config
    
    y_data = data_state.get("y", [])
    if len(y_data) > 0:
        n_classes = len(set(y_data))
        out_dim = n_classes if n_classes > 2 else 1
    else:
        out_dim = 1 

    session_data = {
        "meta": {
            "run_name": run_name,
            "timestamp": original_timestamp, 
            "last_modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            "model_file": model_filename,
            "image_file": image_filename,
        },
        "dataset": ds_config,
        "architecture": {
            "type": "MLP",
            "input_size": getattr(model, "input_size", 0),
            "output_size": out_dim,  
            "layers": layers_data
        },
        "training": {
            "epochs": epochs,
            "lr": lr,
            "seed": int(seed) if seed is not None else 0,
            "optimizer": optim_name,     
            "loss_function": loss_name   
        },
        "history": history,
        "results": {
            "dexire": dexire_res
        }
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=4)
        
    return f"Overwritten: {json_filename}", gr.update(choices=list_saved_runs()), gr.update(visible=False)

def load_run_logic(json_filename):

    if not json_filename:
        return [None] * 19 + [gr.update()] + [None] * 6

    json_path = os.path.join(SAVES_DIR, json_filename)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # A. DATASET
    dc = data["dataset"]
    X, y, feats = None, None, None
    ui_rd, ui_dd, ui_file, ui_tgt, ui_info, fig_data = "Sklearn", None, None, "", "", None
    
    try:
        if dc["source"] == "Sklearn":
            X, y, feats = load_data(dc["name"])
            ui_rd, ui_dd = "Sklearn", dc["name"]
            ui_info = f"Loaded Sklearn Dataset: {dc['name']}"
        elif dc["source"] == "CSV File":
            ui_rd = "CSV File"
            csv_path = dc.get("saved_path")
            if not csv_path or not os.path.exists(csv_path): csv_path = dc.get("path")
            
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                ui_file, ui_tgt = csv_path, dc["target"]
                y = df[ui_tgt].values
                X = df.drop(columns=[ui_tgt]).values
                feats = df.drop(columns=[ui_tgt]).columns.tolist()
                ui_info = f"Reloaded CSV Dataset: {os.path.basename(csv_path)}"
            else:
                ui_info = "Error: CSV file not found."
        
        if y is not None:
            fig_data = plt.figure(figsize=(5,3))
            pd.Series(y).value_counts().plot.pie(autopct='%1.1f%%')
            plt.title("Target Distribution (Reloaded)")
            plt.close(fig_data)

    except Exception as e:
        ui_info = f"Error Data: {str(e)}"

    data_state = {"X": X.tolist() if X is not None else [], "y": y.tolist() if y is not None else [], "features": feats, "config": dc}

    # B. MODEL
    arch = data["architecture"]
    layer_state_list = arch["layers"]
    input_s = arch["input_size"] if arch["input_size"] > 0 else (len(feats) if feats else 0)
    
    out_s = arch.get("output_size", 1)
    
    model = build_mlp_from_layer_df(input_s, layer_state_list, output_size=out_s)
    model.input_size = input_s

    pth_path = os.path.join(MODELS_DIR, data["meta"]["model_file"])
    if os.path.exists(pth_path):
        model.load_state_dict(torch.load(pth_path))
    else:
        ui_info += "\nWarning: Model weights file not found."
    img_struct = Image.open(os.path.join(IMAGE_DIR, data["meta"]["image_file"]))

    # C. HISTORY
    tr = data["training"]
    if X is not None:
        data_dict = prepare_data(X, y, random_seed=tr["seed"])
    else:
        data_dict = None

    hist = data["history"]
    fig_hist = plt.figure(figsize=(10, 4))
    if hist:
        plt.subplot(1, 2, 1); plt.plot(hist['epochs'], hist['train_loss'], label='Train'); plt.plot(hist['epochs'], hist['val_loss'], label='Val'); plt.legend(); plt.title("Loss")
        plt.subplot(1, 2, 2); plt.plot(hist['epochs'], hist['train_acc'], label='Train'); plt.plot(hist['epochs'], hist['val_acc'], label='Val'); plt.legend(); plt.title("Accuracy")
    plt.close(fig_hist)

    # D. XAI
    res = data["results"]
    rules_text = res["dexire"].get("rules", "")
    evo_text = res["dexire"].get("evo", "")
    return (    
        data_state, model, data_dict, hist, #st_data, st_model, st_datadict, st_hist
        ui_rd, ui_dd, ui_file, ui_tgt, ui_info, fig_data, # rd_src, dd_skl, fl_csv, txt_tgt, lbl_data_info, plt_data
        tr["epochs"], tr["lr"], tr["seed"], # nb_ep, nb_lr, nb_seed
        layer_state_list, fig_hist, # st_layer, plt_train
        gr.update(value=rules_text, max_lines=10), # txt_dex
        gr.update(value=evo_text, max_lines=10), # txt_evo
        f"Run loaded: {json_filename}", # lbl_load_info
        img_struct, # img_struct
        gr.update(visible=False),  # btn_save
        True, # st_retrain
        res["dexire"].get("mean",""), # txt_dex_mean
        res["dexire"].get("sum",""), # txt_dex_sum
        res["dexire"].get("rules_number",""), # txt_dex_rules_number
        res["dexire"].get("evo_acuracy",""), # txt_dex_evo_acuracy
        res["dexire"].get("evo_uncov",""), # txt_dex_evo_uncov
        res["dexire"].get("evo_rules_number",""), # txt_dex_evo_rules_number
        res["dexire"].get("evo_rules_length",""), # txt_dex_evo_rules_length
        res["dexire"].get("rules_length",""), # txt_dex_rules_length
        hist['val_acc'][len(hist['val_acc'])-1],hist['val_loss'][len(hist['val_loss'])-1],
        res["dexire"].get("evo_sum",""), # txt_evo_sum
        res["dexire"].get("evo_mean","")  # txt_evo_mean
    )
#txt_dex_rules_number,txt_dex_evo_acuracy,txt_dex_evo_uncov,txt_dex_evo_rules_number