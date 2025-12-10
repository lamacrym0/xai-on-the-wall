import gradio as gr
import pandas as pd
import torch
import numpy as np
import json
import os
import shutil
import matplotlib.pyplot as plt
from datetime import datetime

from src.load_data import load_data
from src.model_builder import build_mlp_from_layer_df
from src.train import train, prepare_data
from explainer.dexire import get_dexire_rules
from explainer.dexire_evo import get_dexire_evo_rules
from explainer.ciu import get_explainer_CIU, get_ciu_instance
from dexire_evo.rule_formatter import format_if_elif_else

# ─────────────────────────────────────────────────────────────────────────────
# FILE & SAVES MANAGER
# ─────────────────────────────────────────────────────────────────────────────

SAVES_DIR = "saves"
MODELS_DIR = os.path.join(SAVES_DIR, "models")
DATA_DIR = os.path.join(SAVES_DIR, "data")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def list_saved_runs():
    if not os.path.exists(SAVES_DIR):
        return []
    files = [f for f in os.listdir(SAVES_DIR) if f.endswith(".json")]
    # Sort by modification time
    files.sort(key=lambda x: os.path.getmtime(os.path.join(SAVES_DIR, x)), reverse=True)
    return files

def save_run_logic(run_name, data_state, layer_config, epochs, lr, seed, model, history, dexire_res, ciu_list):
    if not model or not data_state:
        return "Error: No model or data to save.", gr.update()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{run_name.replace(' ', '_')}_{timestamp}"
    
    # 1. DATASET MANAGEMENT
    ds_config = data_state["config"].copy()
    
    # If source is CSV, copy it to the saves folder for portability
    if ds_config["source"] == "CSV File" and "path" in ds_config:
        original_path = ds_config["path"]
        if os.path.exists(original_path):
            new_csv_name = f"{safe_name}_data.csv"
            new_csv_path = os.path.join(DATA_DIR, new_csv_name)
            shutil.copy(original_path, new_csv_path)
            ds_config["saved_path"] = new_csv_path
            ds_config["original_filename"] = os.path.basename(original_path)

    # 2. MODEL MANAGEMENT
    model_filename = f"{safe_name}.pth"
    model_path = os.path.join(MODELS_DIR, model_filename)
    torch.save(model.state_dict(), model_path)
    
    # 3. JSON METADATA
    layers_data = layer_config.values.tolist() if hasattr(layer_config, "values") else layer_config
    
    session_data = {
        "meta": {
            "run_name": run_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_file": model_filename,
        },
        "dataset": ds_config,
        "architecture": {
            "type": "MLP",
            "input_size": getattr(model, "input_size", 0),
            "layers": layers_data
        },
        "training": {
            "epochs": epochs,
            "lr": lr,
            "seed": seed
        },
        "history": history,
        "results": {
            "dexire": dexire_res,
            "ciu_instances": ciu_list
        }
    }
    
    json_path = os.path.join(SAVES_DIR, f"{safe_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=4)
        
    return f"Saved: {safe_name}", gr.update(choices=list_saved_runs())

def load_run_logic(json_filename):
    if not json_filename:
        return [None] * 20

    json_path = os.path.join(SAVES_DIR, json_filename)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # --- A. DATASET ---
    dc = data["dataset"]
    X, y, feats = None, None, None
    
    ui_rd = "Sklearn"
    ui_dd = None
    ui_file = None
    ui_tgt = ""
    ui_info = ""
    fig_data = None
    
    try:
        if dc["source"] == "Sklearn":
            X, y, feats = load_data(dc["name"])
            ui_rd = "Sklearn"
            ui_dd = dc["name"]
            ui_info = f"Loaded Sklearn Dataset: {dc['name']}"
            
        elif dc["source"] == "CSV File":
            ui_rd = "CSV File"
            csv_path = dc.get("saved_path")
            if not csv_path or not os.path.exists(csv_path):
                csv_path = dc.get("path")
            
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                ui_file = csv_path
                ui_tgt = dc["target"]
                y = df[ui_tgt].values
                X = df.drop(columns=[ui_tgt]).values
                feats = df.drop(columns=[ui_tgt]).columns.tolist()
                ui_info = f"Reloaded CSV Dataset: {os.path.basename(csv_path)}"
            else:
                ui_info = "CSV file not found."
        
        if y is not None:
            fig_data = plt.figure(figsize=(5,3))
            pd.Series(y).value_counts().plot.pie(autopct='%1.1f%%')
            plt.title("Target Distribution (Reloaded)")
            plt.close(fig_data)

    except Exception as e:
        ui_info = f"Data Error: {str(e)}"

    data_state = {
        "X": X.tolist() if X is not None else [], 
        "y": y.tolist() if y is not None else [], 
        "features": feats, 
        "config": dc
    }

    # --- B. MODEL ---
    arch = data["architecture"]
    model_config_df = pd.DataFrame(arch["layers"], columns=["units", "activation"])
    
    input_s = arch["input_size"]
    if input_s == 0 and X is not None:
        input_s = len(feats)

    model = build_mlp_from_layer_df(input_s, model_config_df)
    model.input_size = input_s
    
    pth_filename = data["meta"]["model_file"]
    pth_path = os.path.join(MODELS_DIR, pth_filename)
    if os.path.exists(pth_path):
        model.load_state_dict(torch.load(pth_path))
    else:
        ui_info += "\nModel weights file not found."

    # --- C. HISTORY & SPLITS ---
    tr = data["training"]
    # Re-split data using the same seed to ensure consistency
    if X is not None:
        data_dict = prepare_data(X, y, random_seed=tr["seed"])
    else:
        data_dict = None

    hist = data["history"]
    fig_hist = plt.figure(figsize=(10, 4))
    if hist:
        plt.subplot(1, 2, 1)
        plt.plot(hist['epochs'], hist['train_loss'], label='Train')
        plt.plot(hist['epochs'], hist['val_loss'], label='Val')
        plt.title("Loss")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(hist['epochs'], hist['train_acc'], label='Train')
        plt.plot(hist['epochs'], hist['val_acc'], label='Val')
        plt.title("Accuracy")
        plt.legend()
    plt.close(fig_hist)

    # --- D. XAI ---
    res = data["results"]
    dexire_txt = res["dexire"].get("rules", "")
    dexire_evo_txt = res["dexire"].get("evo", "")
    ciu_list = res.get("ciu_instances", [])

    return (
        data_state, model, data_dict, hist, ciu_list,
        ui_rd, ui_dd, ui_file, ui_tgt, ui_info, fig_data,
        tr["epochs"], tr["lr"], tr["seed"],
        model_config_df,
        fig_hist,
        dexire_txt, dexire_evo_txt,
        ciu_list,
        f"Run loaded: {json_filename}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# UI LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def load_data_ui(source, name, file_obj, target_col):
    if source == "Sklearn":
        X, y, feats = load_data(name)
        cfg = {"source": "Sklearn", "name": name}
    else:
        if file_obj is None: return None, "Missing File", None
        df = pd.read_csv(file_obj.name)
        tgt = target_col.strip() if target_col else df.columns[-1]
        y = df[tgt].values
        X = df.drop(columns=[tgt]).values
        feats = df.drop(columns=[tgt]).columns.tolist()
        cfg = {"source": "CSV File", "path": file_obj.name, "target": tgt}

    fig = plt.figure(figsize=(5,3))
    pd.Series(y).value_counts().plot.pie(autopct='%1.1f%%')
    plt.title("Target Distribution")
    plt.close(fig)
    
    st = {"X": X.tolist(), "y": y.tolist(), "features": feats, "config": cfg}
    return st, f"Loaded: {len(X)} rows", fig

def train_ui(data_st, layers, ep, lr, seed, progress=gr.Progress()):
    if not data_st: return [None] * 9

    X = np.array(data_st["X"])
    y = np.array(data_st["y"])
    
    model = build_mlp_from_layer_df(X.shape[1], layers)
    model.input_size = X.shape[1]
    
    model, d_dict, hist = train(X, y, epochs=int(ep), lr=lr, model=model, random_seed=int(seed), callback_progress=progress)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist['epochs'], hist['train_loss'], label='Train')
    ax[0].plot(hist['epochs'], hist['val_loss'], label='Val')
    ax[0].legend()
    ax[0].set_title("Loss")
    
    ax[1].plot(hist['epochs'], hist['train_acc'], label='Train')
    ax[1].plot(hist['epochs'], hist['val_acc'], label='Val')
    ax[1].legend()
    ax[1].set_title("Accuracy")
    plt.close(fig)
    
    return (
        model,      # st_model
        d_dict,     # st_datadict
        hist,       # st_hist
        fig,        # plt_train
        "",         # out_dex (Empty text)
        "",         # out_evo (Empty text)
        None,       # out_ciu_plot (Empty plot)
        [],         # out_ciu_json (Empty list)
        []          # st_ciu_list (Empty state)
    )

# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="XAI") as demo:
    
    st_data = gr.State(None)
    st_model = gr.State(None)
    st_datadict = gr.State(None)
    st_hist = gr.State(None)
    st_ciu_list = gr.State([])

    gr.Markdown("# XAI")
    
    # MANAGER SECTION
    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            gr.Markdown("### Save Run")
            with gr.Row():
                txt_run_name = gr.Textbox(label="Run Name", value="Run_1", scale=2)
                btn_save = gr.Button("Save", variant="primary", scale=1)
            lbl_save_status = gr.Markdown("")
            
        with gr.Column(scale=2):
            gr.Markdown("### Load Run")
            with gr.Row():
                dd_saves = gr.Dropdown(choices=list_saved_runs(), label="Available Saves", interactive=True, scale=2)
                btn_refresh = gr.Button("Refresh", scale=0)
                btn_load = gr.Button("Load", scale=1)
            lbl_load_info = gr.Markdown("")

    with gr.Tabs():
        # TAB 1: DATASET
        with gr.Tab("1. Dataset"):
            with gr.Row():
                with gr.Column():
                    rd_src = gr.Radio(["Sklearn", "CSV File"], label="Source", value="Sklearn")
                    dd_skl = gr.Dropdown(["Breast Cancer", "Iris", "Wine"], label="Dataset", value="Breast Cancer")
                    fl_csv = gr.File(label="CSV File", visible=False)
                    txt_tgt = gr.Textbox(label="Target Column", visible=False)
                    btn_load_data = gr.Button("Load Data")
                with gr.Column():
                    lbl_data_info = gr.Textbox(label="Info", lines=2)
                    plt_data = gr.Plot(label="Distribution")

        # TAB 2: MODEL
        with gr.Tab("2. Model"):
            with gr.Row():
                with gr.Column():
                    df_layers = gr.Dataframe(headers=["units", "activation"], value=[[16, "relu"], [8, "relu"]], label="Hidden Layers", row_count=(1, "dynamic"))
                    with gr.Row():
                        nb_ep = gr.Number(50, label="Epochs")
                        nb_lr = gr.Number(0.01, label="Learning Rate")
                        nb_seed = gr.Number(42, label="Seed")
                    btn_train = gr.Button("Train Model", variant="primary")
                with gr.Column():
                    plt_train = gr.Plot(label="Loss/Accuracy")

        # TAB 3: EXPLAINABILITY
        with gr.Tab("3. Explainability"):
            with gr.Tabs():
                with gr.Tab("DexiRE (Global)"):
                    btn_dexire = gr.Button("Calculate Global Rules")
                    with gr.Row():
                        out_dex = gr.Textbox(label="Standard Rules", lines=10)
                        out_evo = gr.Textbox(label="Evolutionary Rules", lines=10)
                with gr.Tab("CIU (Local)"):
                    with gr.Row():
                        nb_idx = gr.Number(0, label="Test Instance Index", precision=0)
                        btn_ciu = gr.Button("Explain Instance")
                        btn_ciu_add = gr.Button("Save CIU")
                    out_ciu_plot = gr.Plot()
                    out_ciu_json = gr.JSON(label="Saved CIUs (Session)")

    rd_src.change(lambda x: {dd_skl: gr.update(visible=x=="Sklearn"), fl_csv: gr.update(visible=x!="Sklearn"), txt_tgt: gr.update(visible=x!="Sklearn")}, rd_src, [dd_skl, fl_csv, txt_tgt])
    
    btn_refresh.click(lambda: gr.update(choices=list_saved_runs()), outputs=dd_saves)

    btn_load_data.click(load_data_ui, inputs=[rd_src, dd_skl, fl_csv, txt_tgt], outputs=[st_data, lbl_data_info, plt_data])

    # Connect train button to reset XAI outputs
    btn_train.click(
        train_ui, 
        inputs=[st_data, df_layers, nb_ep, nb_lr, nb_seed], 
        outputs=[
            st_model, st_datadict, st_hist, plt_train, # Results
            out_dex, out_evo, out_ciu_plot, out_ciu_json, st_ciu_list # Resets
        ]
    )

    def wrap_save(name, d_st, lay, ep, lr, seed, mod, hist, dx, ciu, dx_evo):
        dx_res = {"rules": dx, "evo": dx_evo}
        return save_run_logic(name, d_st, lay, ep, lr, seed, mod, hist, dx_res, ciu)

    btn_save.click(
        wrap_save, 
        inputs=[txt_run_name, st_data, df_layers, nb_ep, nb_lr, nb_seed, st_model, st_hist, out_dex, out_ciu_json, out_evo],
        outputs=[lbl_save_status, dd_saves]
    )

    btn_load.click(
        load_run_logic,
        inputs=[dd_saves],
        outputs=[
            st_data, st_model, st_datadict, st_hist, st_ciu_list,
            rd_src, dd_skl, fl_csv, txt_tgt, lbl_data_info, plt_data, # Data UI
            nb_ep, nb_lr, nb_seed, df_layers, plt_train, # Model UI
            out_dex, out_evo, out_ciu_json, # XAI UI
            lbl_load_info
        ]
    )

    def run_dex(mod, d_dict, d_st):
        if not mod: return "", ""
        feats = d_st["features"]
        r, _ = get_dexire_rules(mod, d_dict, feats)
        best, _, _, eng = get_dexire_evo_rules(feats, mod, d_dict)
        re = format_if_elif_else(best, feats, eng.operator_set)
        return r, re
    btn_dexire.click(run_dex, [st_model, st_datadict, st_data], [out_dex, out_evo])

    def run_ciu(idx, mod, d_dict, d_st):
        if not mod: return None
        feats = d_st["features"]
        ciu = get_explainer_CIU(mod, d_dict, ["Class0", "Class1"], feats)
        
        X_test_df = pd.DataFrame(d_dict["X_test"], columns=feats)
        instance = X_test_df.iloc[[int(idx)]]
        
        res = get_ciu_instance(ciu, instance)
        
        ciu_plot_out = ciu.plot_ciu(res, figsize=(9, 6))
        ciu_plot_out.subplots_adjust(left=0.25)
        ax = ciu_plot_out.axes[0] 
        ax.set_xlim(0, 0.25)
        plt.close(ciu_plot_out)
        
        return ciu_plot_out
        
    btn_ciu.click(run_ciu, [nb_idx, st_model, st_datadict, st_data], out_ciu_plot)

    btn_ciu_add.click(lambda idx, lst: (lst + [{"idx": int(idx)}], lst + [{"idx": int(idx)}]), [nb_idx, st_ciu_list], [st_ciu_list, out_ciu_json])

if __name__ == "__main__":
    demo.launch()