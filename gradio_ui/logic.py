import matplotlib.patches as patches
import matplotlib.pyplot as plt
import gradio as gr
import torch
import io
import pandas as pd
import numpy as np
import random
from PIL import Image
from explainer.dexire import make_sympy_safe_names

from src.model_builder import build_mlp_from_layer_df
from src.train import train
from src.load_data import load_data
from explainer.ciu import get_explainer_CIU, get_ciu_instance

# ─────────────────────────────────────────────────────────────────────────────
# UI LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def visualize_model_matplotlib(model):
    layers = list(model.children())
    n_layers = len(layers)
    
    fig, ax = plt.subplots(figsize=(4, max(3, n_layers * 0.8)))
    ax.axis('off')
    
    x_center = 0.5
    y_top = 0.95
    y_bottom = 0.05
    available_height = y_top - y_bottom
    y_step = available_height / n_layers if n_layers > 0 else 0.5
    
    box_w = 0.4        
    box_h = min(0.08, y_step * 0.5) 
    fontsize = 9       
    
    current_y = y_top
    
    for layer in layers:
        layer_type = type(layer).__name__
        details = ""
        color = "#e3f2fd"
        
        if isinstance(layer, torch.nn.Linear):
            details = f"In: {layer.in_features} Out: {layer.out_features}"
            color = "#bbdefb"
        elif isinstance(layer, (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh)):
            color = "#f3e5f5"
        
        rect = patches.FancyBboxPatch(
            (x_center - box_w/2, current_y - box_h),
            box_w, box_h,
            boxstyle="round,pad=0.02", 
            edgecolor="#1565c0",
            facecolor=color,
            linewidth=1.0 
        )
        ax.add_patch(rect)
        
        label = f"{layer_type}"
        if details:
            label += f"\n{details}"
            
        ax.text(x_center, current_y - box_h/2, label, 
                ha='center', va='center', fontsize=fontsize, fontweight='bold', color="#0d47a1")
        
        current_y -= y_step
        
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120) 
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def format_dexire_output(rules_str):
    if not rules_str or rules_str == "[]":
        return "No rules found."
    
    content = str(rules_str).strip("[]")
    if not content:
        return "No rules."
        
    rules = content.split(", IF")
    
    formatted = []
    for i, r in enumerate(rules):
        rule_text = r.strip()
        if i > 0:
            rule_text = "IF " + rule_text
            
        rule_text = rule_text.replace(" AND ", "\n    AND ")
        rule_text = rule_text.replace(" THEN ", "\n  THEN ")
        
        formatted.append(rule_text)
        
    return "\n\n".join(formatted)
def count_mean_features(ruleset, feature_names):
    feature_means = {}
    feature_counts = {}

    for rule in ruleset:
        for f_idx, threshold, op_idx in rule[:-1]:
            
            if 0 <= f_idx < len(feature_names):
                name = feature_names[f_idx]
                
                value = abs(float(threshold))
                
                if name in feature_counts:
                    feature_counts[name] += 1
                    feature_means[name] += value
                else:
                    feature_counts[name] = 1
                    feature_means[name] = value

    for feature, total_val in feature_means.items():
        count = feature_counts[feature]
        if count > 0:
            feature_means[feature] = total_val / count

    sorted_means = sorted(feature_means.items(), key=lambda x: x[1], reverse=True)
    sorted_counts = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

    return sorted_means, sorted_counts

def format_dexire_evo_output(ruleset, feature_names, operator_set):
    ordered = sorted(ruleset, key=lambda r: len(r) - 1, reverse=True)
    lines = []
    for r in ordered:
        conds = "\n    AND ".join(
            f"{feature_names[f]} {operator_set.get(op_idx).symbol} {thr:.2f}"
            for f, thr, op_idx in r[:-1]
        )
        prefix = "IF "
        if conds:
            lines.append(f"{prefix} {conds}\n  THEN {r[-1][1]}\n")
        else:
            lines.append(f"{prefix}\n  THEN {r[-1][1]}\n")
    return "\n".join(lines)

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
    return st, f"Loaded: {len(X)} rows", fig, gr.update(visible=False),gr.update(visible=True),gr.update(value=None),gr.update(value=None),"",""

def train_ui(data_st, layers_list, ep, lr, seed, optim_name, loss_name, use_wandb):
    if not data_st: return [None] * 7 + [seed]

    
    generated_seed = int(seed) if seed is not None else random.randint(0, 10000)
    
    X = np.array(data_st["X"])
    y = np.array(data_st["y"])
    
    # 1. Dynamic output size calculation
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    output_size = num_classes if num_classes > 2 else 1
    
    # 2. Build model
    model = build_mlp_from_layer_df(X.shape[1], layers_list, output_size=output_size)
    model.input_size = X.shape[1]
    
    # 3. Train
    model, d_dict, hist = train(
        X, y, 
        epochs=int(ep), 
        lr=lr, 
        model=model, 
        random_seed=generated_seed,
        optimizer_name=optim_name,
        loss_name=loss_name,
        use_wandb=use_wandb,
    )
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist['epochs'], hist['train_loss'], label='Train'); ax[0].plot(hist['epochs'], hist['val_loss'], label='Val'); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(hist['epochs'], hist['train_acc'], label='Train'); ax[1].plot(hist['epochs'], hist['val_acc'], label='Val'); ax[1].legend(); ax[1].set_title("Accuracy")
    plt.close(fig)


    try:
        viz_path = visualize_model_matplotlib(model)
    except Exception as e:
        viz_path = None
    return (
        model, d_dict, hist, fig, viz_path, 
        "", "", None, 
        generated_seed,
        gr.update(visible=True),
        "",None,"","",gr.update(visible=True),"","","","",hist['val_acc'][len(hist['val_acc'])-1],hist['val_loss'][len(hist['val_loss'])-1]
    )

def run_ciu(idx, mod, d_dict, d_st):
    if not mod: return None
    feats = d_st["features"]
    
    last_linear = None
    for layer in reversed(mod):
        if isinstance(layer, torch.nn.Linear):
            last_linear = layer
            break
            
    if last_linear is None:
        return None 

    out_dim = last_linear.out_features

    if out_dim == 1:
        classes = ["Class0", "Class1"]
    else:
        classes = [f"Class{i}" for i in range(out_dim)]

    ciu = get_explainer_CIU(mod, d_dict, classes, feats)
    
    X_test_df = pd.DataFrame(d_dict["X_test"], columns=feats)
    
    if int(idx) >= len(X_test_df): 
        return None
    
    instance = X_test_df.iloc[[int(idx)]]
    try:
        res = get_ciu_instance(ciu, instance)
    
        fig_influance = ciu.plot_influence(res,figsize=(9, 6)) 
        fig = ciu.plot_ciu(res, figsize=(9, 6))
        fig.subplots_adjust(left=0.25)
        fig_influance.subplots_adjust(left=0.25)
        ax = fig_influance.axes[0]
        ax.set_xlim(-0.1, 0.1)
        plt.close(fig_influance)
        ax = fig.axes[0]
        ax.set_xlim(0, 0.25) 
        plt.close(fig)
        return fig, fig_influance
    except Exception as e:
        return None,None