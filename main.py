import pandas as pd 
import gradio as gr
import datetime

from explainer.dexire import get_dexire_rules
from explainer.ciu import get_explainer_CIU, get_ciu_instance
from src.train import train
from explainer.dexire_evo import get_dexire_evo_rules
from dexire_evo.rule_formatter import format_if_elif_else
from src.model_builder import build_mlp_from_layer_df
from src.load_data import load_data


# ─────────────────────────────────────────────
# Explanation for a specific instance CIU
# ─────────────────────────────────────────────

def run_explanations_for_specific_instance(test_index, model, data, feature_names):
    # test_index comes as float sometimes, ensure int and clipped later if needed
    idx = int(test_index)

    CIU_model = get_explainer_CIU(
        model,
        data,
        output_names=["malignant", "benign"],
        feature_names=feature_names,
    )

    X_test_df = pd.DataFrame(data["X_test"], columns=feature_names)

    # Basic safety: clamp index
    idx = max(0, min(idx, len(X_test_df) - 1))

    res = get_ciu_instance(CIU_model, X_test_df.iloc[[idx]])
    ciu_plot_out = CIU_model.plot_ciu(res, figsize=(9, 6))

    return ciu_plot_out


# ─────────────────────────────────────────────
# Main training + DexiRE / DexiRE-Evo
# ─────────────────────────────────────────────

def run_explanations(
    dataset_mode,
    dataset_file,
    dataset_choice,
    model_mode,
    layer_config,
    model_state_file,
    save_trained_model,
    learning_rate,
    epochs,
):
    print("Layer config:", layer_config)

    # --- Load data ---
    if dataset_mode == "Sklearn Dataset":
        X, y, feature_names = load_data(dataset_choice)
    else:
        X, y, feature_names = load_data(dataset_file)
        feature_names = X.columns.tolist()

    input_size = X.shape[1]

    # --- Build model architecture from layer_config ---
    model_architecture = build_mlp_from_layer_df(input_size, layer_config)

    # --- Train or load model ---
    if model_mode == "Load existing model":
        # model_state_file is a gr.File object; you may want model_state_file.name inside train()
        model, data = train(
            X=X,
            y=y,
            model_state=model_state_file,
            model=model_architecture,
        )
    else:
        model, data = train(
            X=X,
            y=y,
            model=model_architecture,
            save=save_trained_model,
            lr=learning_rate,
            epochs=epochs,
        )

    model.eval()

    # --- DexiRE ---
    dexire_out = get_dexire_rules(model, data, feature_names=feature_names)

    # --- DexiRE-Evo ---
    best, test_acc, uncov_te, engine = get_dexire_evo_rules(
        feature_names, model, data
    )
    result_str = ""
    result_str += "Rules in IF–ELIF–ELSE form (GA):\n"
    result_str += "========================\n"
    result_str += format_if_elif_else(best, feature_names, engine.operator_set) + "\n"
    result_str += "========================\n"
    result_str += f"Fidelity (train vs model): {best.fitness.values[0]:.3f}\n"
    result_str += f"# Predicates             : {best.fitness.values[1]}\n"
    result_str += f"Uncovered (train)        : {best.fitness.values[2]}\n"
    result_str += f"Test accuracy (matched)  : {test_acc:.3f} | Uncov test: {uncov_te}\n"
    dexire_evo_out = result_str

    return dexire_out, dexire_evo_out, model, data, feature_names


# ─────────────────────────────────────────────
# Save results helper
# ─────────────────────────────────────────────

def save_results(dexire_text, dexire_evo_text):
    """
    Save DexiRE and DexiRE-Evo outputs to a text file and return its path.
    Gradio File output will use this for download.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"xai_results_{timestamp}.txt"

    content = ""
    content += "=== DexiRE rules ===\n"
    content += (dexire_text or "") + "\n\n"
    content += "=== DexiRE-Evo rules & metrics ===\n"
    content += (dexire_evo_text or "") + "\n"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    return filename  # gr.File will serve this


# ─────────────────────────────────────────────
# Visibility toggle helpers
# ─────────────────────────────────────────────

def toggle_dataset_file(dataset_mode):
    return (
        gr.update(visible=(dataset_mode == "Upload CSV")),      # dataset_file
        gr.update(visible=(dataset_mode == "Sklearn Dataset")), # dataset_choice
    )


def toggle_model_widgets(model_mode):
    load_mode = (model_mode == "Load existing model")
    return (
        gr.update(visible=load_mode),       # model_state_file
        gr.update(visible=not load_mode),   # train_cfg_group
        gr.update(visible=not load_mode),   # save_trained_model
        gr.update(visible=not load_mode),   # learning_rate
        gr.update(visible=not load_mode),   # epochs
    )


# ─────────────────────────────────────────────
# Gradio Interface
# ─────────────────────────────────────────────

with gr.Blocks(title="XAI on the Wall") as main:
    gr.Markdown("## XAI on the Wall – CIU, DexiRE & DexiRE-Evo")

    # States to store model, data and feature_names
    model_state_gr = gr.State()
    data_state_gr = gr.State()
    feature_names_gr = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            # ── Dataset section ──
            gr.Markdown("### Dataset")

            dataset_mode = gr.Radio(
                ["Sklearn Dataset", "Upload CSV"],
                value="Sklearn Dataset",
                label="Dataset source",
            )

            dataset_choice = gr.Radio(
                ["Iris", "Wine Quality", "Breast Cancer"],
                value="Breast Cancer",
                label="Which demo dataset?",
            )

            dataset_file = gr.File(
                label="Upload dataset (CSV, last column = target)",
                file_types=[".csv"],
                interactive=True,
                visible=False,
            )

            # ── Model section ──
            gr.Markdown("### Model")

            model_mode = gr.Radio(
                ["Train new model", "Load existing model"],
                value="Train new model",
                label="Model source",
            )

            model_state_file = gr.File(
                label="Model state (.pth) (used if 'Load existing model')",
                file_types=[".pth", ".pt"],
                interactive=True,
                visible=False,
            )

            with gr.Group(visible=False) as train_cfg_group:
                save_trained_model = gr.Checkbox(
                    value=False,
                    label="Save trained model to disk",
                )
                learning_rate = gr.Number(
                    value=0.001,
                    precision=3,
                    label="Learning rate",
                )
                epochs = gr.Number(
                    value=50,
                    precision=0,
                    label="Epochs",
                )

                layer_config = gr.Dataframe(
                    headers=["units", "activation"],
                    datatype=["number", "str"],
                    row_count=(2, "dynamic"),
                    value=[
                        [16, "relu"],
                        [8, "relu"],
                    ],
                    label="Layers (add/remove rows):",
                )

            run_btn = gr.Button("Train / Load")

            test_index = gr.Number(
                value=0,
                precision=0,
                label="Index of test instance to explain (0-based)",
            )
            run_instance_btn = gr.Button("Explain instance")

        # ── Outputs section ──
        with gr.Column(scale=2):
            gr.Markdown("### DexiRE")
            dexire_out = gr.Textbox(
                label="DexiRE rules",
                lines=10,
            )

            gr.Markdown("### DexiRE-Evo")
            dexire_evo_out = gr.Textbox(
                label="DexiRE-Evo rules & metrics",
                lines=10,
            )

            # Save/download UI
            save_btn = gr.Button("Save results to file")
            saved_file = gr.File(label="Download saved results")

            ciu_plot_out = gr.Plot(label="CIU plot")

    # ── Hook up visibility toggles ──

    dataset_mode.change(
        fn=toggle_dataset_file,
        inputs=dataset_mode,
        outputs=[dataset_file, dataset_choice],
    )

    model_mode.change(
        fn=toggle_model_widgets,
        inputs=model_mode,
        outputs=[model_state_file, train_cfg_group, save_trained_model, learning_rate, epochs],
    )

    main.load(
        fn=toggle_model_widgets,
        inputs=model_mode,
        outputs=[model_state_file, train_cfg_group, save_trained_model, learning_rate, epochs],
    )

    # ── Main run button: training + DexiRE / DexiRE-Evo ──
    run_btn.click(
        fn=run_explanations,
        inputs=[
            dataset_mode,
            dataset_file,
            dataset_choice,
            model_mode,
            layer_config,
            model_state_file,
            save_trained_model,
            learning_rate,
            epochs,
        ],
        outputs=[
            dexire_out,
            dexire_evo_out,
            model_state_gr,
            data_state_gr,
            feature_names_gr,
        ],
    )

    # ── Explain specific instance ──
    run_instance_btn.click(
        fn=run_explanations_for_specific_instance,
        inputs=[
            test_index,
            model_state_gr,
            data_state_gr,
            feature_names_gr,
        ],
        outputs=[ciu_plot_out],
    )

    # ── Save results button ──
    save_btn.click(
        fn=save_results,
        inputs=[dexire_out, dexire_evo_out],
        outputs=[saved_file],
    )

main.launch()  # add share=True to have a public server
