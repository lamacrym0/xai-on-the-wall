import gradio as gr

from explainer.dexire import get_dexire_rules
from explainer.dexire_evo import get_dexire_evo_rules
from gradio_ui.logic import format_dexire_output, format_dexire_evo_output, load_data_ui, train_ui, run_ciu
from gradio_ui.save_manager import list_saved_runs, save_run_logic, load_run_logic, overwrite_run_logic

# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="XAI Workstation") as demo:
    st_data, st_model, st_datadict, st_hist,st_retrain = gr.State(None), gr.State(None), gr.State(None), gr.State(None),gr.State(True)

    gr.Markdown("# XAI Workstation")
    
    # MANAGER PANEL
    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            gr.Markdown("### Save Run")
            with gr.Row():
                txt_run_name = gr.Textbox(label="Run Name", value="Run_1", scale=2)
                btn_save_new = gr.Button("Save New", variant="primary", scale=1)
            lbl_save_status = gr.Markdown("")
        with gr.Column(scale=2):
            gr.Markdown("### Load Run")
            with gr.Row():
                dd_saves = gr.Dropdown(choices=list_saved_runs(), label="Available Saves", interactive=True, scale=2)
                btn_refresh, btn_load = gr.Button("Refresh", scale=0), gr.Button("Load", scale=1)
            with gr.Row():
                lbl_load_info = gr.Markdown("")
                btn_save = gr.Button("Save", visible=True) 

    demo.load(lambda: gr.update(visible=False), None, btn_save)


    with gr.Tabs():
        # TAB 1: DATA
        with gr.Tab("1. Dataset"):
            with gr.Row():
                with gr.Column():
                    rd_src = gr.Radio(["Sklearn", "CSV File"], label="Source", value="Sklearn")
                    dd_skl = gr.Dropdown(["Breast Cancer", "Iris", "Wine"], label="Dataset", value="Breast Cancer")
                    txt_tgt = gr.Textbox(label="Target Column", visible=False)
                    fl_csv = gr.File(label="CSV File", visible=False)
                    btn_load_data = gr.Button("Load Data")
                with gr.Column():
                    lbl_data_info = gr.Textbox(label="Info", lines=2)
                    plt_data = gr.Plot(label="Distribution")

        # TAB 2: MODEL
        with gr.Tab("2. Model"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Architecture Designer")
                    
                    layer_state = gr.State([[16, "relu"], [8, "relu"]])

                    @gr.render(inputs=layer_state)
                    def render_layers(layers):
                        if not layers:
                            gr.Markdown("*Aucune couche cachée définie.*")
                            return
                        
                        for i, (units, act) in enumerate(layers):
                            with gr.Row(variant="panel"):
                                nb = gr.Number(
                                    value=units, 
                                    label=f"Layer {i+1} Units", 
                                    interactive=True  
                                )
                                
                                dd = gr.Dropdown(
                                    ["relu", "tanh", "sigmoid", "elu", "leaky_relu"], 
                                    value=act, 
                                    label="Activation", 
                                    interactive=True  
                                )
                                
                                btn_del = gr.Button("Delete", size="sm", scale=0)

                                def update_units(new_val, idx=i):
                                    layers[idx][0] = int(new_val)
                                
                                nb.input(update_units, inputs=nb, outputs=None) 
                                nb.input(lambda: False, None, st_retrain)

                                def update_act(new_val, idx=i):
                                    layers[idx][1] = new_val
                                    
                                dd.change(update_act, inputs=dd, outputs=None)
                                dd.change(lambda: False, None, st_retrain)

                                def delete_layer(idx=i):
                                    layers.pop(idx)
                                    return layers
                                    
                                btn_del.click(delete_layer, None, layer_state)
                    btn_add_layer = gr.Button("Add Layer", variant="secondary")

                    def add_new_layer(current_layers):
                        return current_layers + [[16, "relu"]]
                    
                    btn_add_layer.click(add_new_layer, inputs=layer_state, outputs=layer_state)
                    btn_add_layer.click(lambda: False, None, st_retrain)
                    
                    with gr.Row():
                        dd_optim = gr.Dropdown(["Adam", "SGD", "RMSprop"], label="Optimizer", value="Adam")
                        dd_loss = gr.Dropdown(["Auto", "BCELoss", "CrossEntropyLoss", "MSELoss"], label="Loss Function", value="Auto")

                    with gr.Row():
                        nb_ep = gr.Number(50, label="Epochs")
                        nb_lr = gr.Number(0.001, label="Learning Rate")
                        nb_seed = gr.Number(42, label="Seed (Empty = Random)", precision=0)
                    
                    cb_wandb = gr.Checkbox(label="Track with WandB", value=False)
                    
                    btn_train = gr.Button("Train Model", variant="primary")
                with gr.Column():
                    plt_train = gr.Plot(label="Loss/Accuracy")
                    img_struct = gr.Image(label="Model Architecture", type="filepath")

        # TAB 3: XAI
        with gr.Tab("3. Explainability"):
            btn_dexire = gr.Button("Calculate Global Rules")
            with gr.Row():
                out_dex = gr.Textbox(label="Standard Rules", lines=10)
                out_evo = gr.Textbox(label="Evolutionary Rules", lines=10)
            with gr.Row():
                nb_idx = gr.Number(0, label="Index", precision=0)
                btn_ciu = gr.Button("Explain Instance")
            out_ciu_plot = gr.Plot()

    # EVENTS / WIRING
    rd_src.change(lambda x: {dd_skl: gr.update(visible=x=="Sklearn"), fl_csv: gr.update(visible=x!="Sklearn"), txt_tgt: gr.update(visible=x!="Sklearn")}, rd_src, [dd_skl, fl_csv, txt_tgt])
    btn_refresh.click(lambda: gr.update(choices=list_saved_runs()), outputs=dd_saves)
    
    btn_load_data.click(load_data_ui, inputs=[rd_src, dd_skl, fl_csv, txt_tgt], outputs=[st_data, lbl_data_info, plt_data,btn_save])

    # TRAIN EVENT
    btn_train.click(
        train_ui, 
        inputs=[st_data, layer_state, nb_ep, nb_lr, nb_seed, dd_optim, dd_loss, cb_wandb], 
        outputs=[st_model, st_datadict, st_hist, plt_train, img_struct, out_dex, out_evo, out_ciu_plot, nb_seed,btn_save,lbl_load_info]
    )

    # SAVE EVENT
    def wrap_save(name, d_st, lay, ep, lr, seed, mod, hist, dx, dx_evo, opt, loss,img_struct):
        dx_res = {"rules": dx, "evo": dx_evo}
        return save_run_logic(name, d_st, lay, ep, lr, seed, mod, hist, dx_res, opt, loss,img_struct)

    btn_save_new.click(
        wrap_save, 
        inputs=[txt_run_name, st_data, layer_state, nb_ep, nb_lr, nb_seed, st_model, st_hist, out_dex, out_evo, dd_optim, dd_loss,img_struct],
        outputs=[lbl_save_status, dd_saves,btn_save]
    )

    def wrap_overwrite(selected_save, d_st, lay, ep, lr, seed, mod, hist, dx, dx_evo, opt, loss,img_struct,st_retrain):
        if not selected_save:
            return "Error: No save selected.", gr.update()
        if(not st_retrain):
            return "Error: You changed the model architecture. Please retrain the model before overwriting the save or reload the save to get the right model version.", gr.update()
        
        return overwrite_run_logic(selected_save, d_st, lay, ep, lr, seed, mod, hist, {"rules": dx, "evo": dx_evo}, opt, loss,img_struct)
    
    btn_save.click(
        wrap_overwrite,
        inputs=[dd_saves, st_data, layer_state, nb_ep, nb_lr, nb_seed, st_model, st_hist, out_dex, out_evo, dd_optim, dd_loss,img_struct,st_retrain],
        outputs=[lbl_save_status, dd_saves,btn_save]
    )

    # LOAD EVENT
    btn_load.click(
        load_run_logic,
        inputs=[dd_saves],
        outputs=[st_data, st_model, st_datadict, st_hist, rd_src, dd_skl, fl_csv, txt_tgt, lbl_data_info, plt_data, nb_ep, nb_lr, nb_seed, layer_state, plt_train, out_dex, out_evo, lbl_load_info,img_struct,btn_save,st_retrain,btn_save]
    )

    def run_dex(mod, d_dict, d_st):
        if not mod: return "", ""
        feats = d_st["features"]
        
        r, _ = get_dexire_rules(mod, d_dict, feats)
        
        r_formatted = format_dexire_output(r)
        
        best, test_acc, uncov_te, eng = get_dexire_evo_rules(feats, mod, d_dict)
        r_evo = format_dexire_evo_output(best, feats, eng.operator_set)

        return r_formatted, r_evo,gr.update(visible=True)
    btn_dexire.click(run_dex, [st_model, st_datadict, st_data], [out_dex, out_evo,btn_save])

    btn_ciu.click(run_ciu, [nb_idx, st_model, st_datadict, st_data], out_ciu_plot)

if __name__ == "__main__":
    demo.launch()