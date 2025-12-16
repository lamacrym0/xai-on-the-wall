import gradio as gr

from explainer.dexire import get_dexire_rules
from explainer.dexire_evo import get_dexire_evo_rules
from gradio_ui.logic import format_dexire_output, format_dexire_evo_output, load_data_ui, train_ui, run_ciu, count_mean_features
from gradio_ui.save_manager import list_saved_runs, save_run_logic, load_run_logic, overwrite_run_logic

# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE
# The base of the graphical part of this interface was generated with the help of ChatGPT.
# dd = Dropdown
# nb = NumberBox
# btn = Button
# lbl = Label
# txt = Textbox
# plt = Plot
# img = Image
# cb = Checkbox
# rd = Radio
# st = State
# ─────────────────────────────────────────────────────────────────────────────

css = """
.code-scroll .cm-scroller { 
    height: 200px !important; 
    overflow-y: auto !important; 
}
"""
with gr.Blocks(title="XAI Workstation") as demo:
    st_data, st_model, st_datadict, st_hist,st_retrain = gr.State(None), gr.State(None), gr.State(None), gr.State(None),gr.State(True)

    gr.Markdown("# XAI Workstation")
    
    # MANAGER PANEL
    with gr.Row(variant="panel"):
        with gr.Column():
            gr.Markdown("### Load Result State")
            dd_saves = gr.Dropdown(choices=list_saved_runs(), label="Available States", interactive=True, scale=2)

        with gr.Column():
            with gr.Row():
                with gr.Row():
                    btn_refresh, btn_load = gr.Button("Refresh", scale=0), gr.Button("Load", scale=1)
            with gr.Row():
                btn_save = gr.Button("Overwrite State", visible=True) 

            lbl_load_info = gr.Markdown("")
                
    demo.load(lambda: gr.update(visible=False), None, btn_save)

    with gr.Tabs() :
        # TAB 1: DATA
        with gr.Tab("1. Dataset") as tab_data:
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
        with gr.Tab("2. Model",visible=False) as tab_model:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Architecture Designer")
                    
                    st_layer = gr.State([[16, "relu"], [8, "relu"]])

                    @gr.render(inputs=st_layer)
                    def render_layers(layers):
                        if not layers:
                            gr.Markdown("No hidden layer defined")
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
                                    
                                btn_del.click(delete_layer, None, st_layer)
                    btn_add_layer = gr.Button("Add Layer", variant="secondary")

                    def add_new_layer(current_layers):
                        return current_layers + [[16, "relu"]]
                    
                    btn_add_layer.click(add_new_layer, inputs=st_layer, outputs=st_layer)
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
                    with gr.Row():
                        txt_loss = gr.Textbox(label="Final Loss", lines=1, interactive=False)
                        txt_acc = gr.Textbox(label="Final Accuracy", lines=1, interactive=False)
                    img_struct = gr.Image(label="Model Architecture", type="filepath")
                    
        
        # TAB 3: XAI
        with gr.Tab("3. Explainer", visible=False) as tab_xai:
            btn_dexire = gr.Button("Calculate Global Rules")
            with gr.Row():
                txt_dex = gr.Code(label="Standard Rules", lines=10,max_lines=10, interactive=False, elem_classes="code-scroll")
                txt_evo = gr.Code(label="Evo Rules", lines=10,max_lines=10, interactive=False, elem_classes="code-scroll")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### DEXiRE Summary Statistics")
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                txt_dex_rules_number = gr.Textbox(label="Number of Extracted Rules", lines=1,max_lines=10000, interactive=False)
                                txt_dex_rules_length = gr.Textbox(label="Average Rule Length", lines=1,max_lines=10000, interactive=False)
                        with gr.Column():
                            with gr.Row():
                                txt_dex_evo_rules_number = gr.Textbox(label="Number of Extracted Rules with Evolution", lines=1,max_lines=10000, interactive=False)
                                txt_dex_evo_rules_length = gr.Textbox(label="Average Rule Length with Evolution", lines=1,max_lines=10000, interactive=False)
                    with gr.Row():
                        txt_dex_sum = gr.Code(label="Features Count in Rules", lines=10,max_lines=10, interactive=False, elem_classes="code-scroll")
                        txt_dex_mean = gr.Code(label="Features Conditions Means", lines=10,max_lines=10, interactive=False, elem_classes="code-scroll")
                        txt_evo_sum = gr.Code(label="DEXiRE Evo Features Count in Rules", lines=10,max_lines=10, interactive=False, elem_classes="code-scroll")
                        txt_evo_mean = gr.Code(label="DEXiRE Evo Features Conditions Means", lines=10,max_lines=10, interactive=False, elem_classes="code-scroll")
                    with gr.Row():
                        txt_dex_evo_uncov = gr.Textbox(label="DEXiRE Evo Test Uncovered Instances", lines=1,max_lines=10000, interactive=False)
                        txt_dex_evo_acuracy = gr.Textbox(label="DEXiRE Evo Accuracy", lines=1,max_lines=10000, interactive=False)
            gr.Markdown("---")

            gr.Markdown("### Explain Individual Predictions")

            with gr.Row():
                nb_idx = gr.Number(0, label="Index", precision=0)
                btn_ciu = gr.Button("Explain Instance")
            with gr.Row():
                plt_ciu = gr.Plot(label="CIU Explanation")
                plt_ciu_influance = gr.Plot(label="CIU Influence")
    
    # SAVE PANEL
    with gr.Column(variant="panel", visible=False) as save_panel:
        title_save = gr.Markdown("### Save Result State")
        txt_run_name = gr.Textbox(label="Result State Name", value="Result_State_1", scale=2)
        btn_save_new = gr.Button("Save New State", variant="primary", scale=1)
        lbl_save_status = gr.Markdown("")
        
    # EVENTS / WIRING
    rd_src.change(lambda x: {dd_skl: gr.update(visible=x=="Sklearn"), fl_csv: gr.update(visible=x!="Sklearn"), txt_tgt: gr.update(visible=x!="Sklearn")}, rd_src, [dd_skl, fl_csv, txt_tgt])
    btn_refresh.click(lambda: gr.update(choices=list_saved_runs()), outputs=dd_saves)
    
    btn_load_data.click(
        load_data_ui, 
        inputs=[rd_src, dd_skl, fl_csv, txt_tgt],
        outputs=[st_data, lbl_data_info, plt_data,btn_save,tab_model,plt_train,img_struct,txt_loss,txt_acc]
    ).then(lambda : [gr.update(visible=False)]*5, outputs=[save_panel,title_save,txt_run_name,lbl_save_status,btn_save_new]).then(lambda : [""]*12, outputs=[txt_dex,txt_evo,txt_dex_rules_length,txt_dex_rules_number,txt_dex_evo_rules_number,txt_dex_evo_rules_length,txt_dex_sum,txt_dex_mean,txt_evo_sum,txt_evo_mean,txt_dex_evo_uncov,txt_dex_evo_acuracy])

    # TRAIN EVENT
    btn_train.click(
        train_ui, 
        inputs=[st_data, st_layer, nb_ep, nb_lr, nb_seed, dd_optim, dd_loss, cb_wandb], 
        outputs=[st_model, st_datadict, st_hist, plt_train, img_struct, txt_dex, txt_evo, plt_ciu, nb_seed,btn_save,lbl_load_info,
                 plt_ciu_influance,txt_dex_mean,txt_dex_sum,tab_xai,txt_dex_rules_number,txt_dex_evo_acuracy,txt_dex_evo_uncov,
                 txt_dex_evo_rules_number,txt_acc,txt_loss]
    ).then(lambda : [gr.update(visible=True)]*5, outputs=[save_panel,title_save,txt_run_name,lbl_save_status,btn_save_new])
    

    # SAVE EVENT
    def wrap_save(name, st_data, st_layer, nb_ep, nb_lr, nb_seed, st_model, st_hist, txt_dex, txt_evo, dd_optim, dd_loss,img_struct,txt_dex_mean,txt_dex_sum,txt_dex_rules_number,txt_dex_evo_acuracy,txt_dex_evo_uncov,txt_dex_evo_rules_number,txt_dex_evo_rules_length,txt_dex_rules_length,txt_evo_sum,txt_evo_mean):
        dex_res = {"rules": txt_dex,"rules_number": txt_dex_rules_number, "evo": txt_evo,"mean": txt_dex_mean,"sum": txt_dex_sum,"evo_uncov": txt_dex_evo_uncov,"evo_rules_number": txt_dex_evo_rules_number, "evo_acuracy": txt_dex_evo_acuracy,"evo_rules_length": txt_dex_evo_rules_length,"rules_length": txt_dex_rules_length,"evo_sum": txt_evo_sum,"evo_mean": txt_evo_mean}
        return save_run_logic(name, st_data, st_layer, nb_ep, nb_lr, nb_seed, st_model, st_hist, dex_res, dd_optim, dd_loss,img_struct)

    btn_save_new.click(
        wrap_save, 
        inputs=[txt_run_name, st_data, st_layer, nb_ep, nb_lr, nb_seed, st_model, st_hist, txt_dex, txt_evo, dd_optim, dd_loss,img_struct,txt_dex_mean,txt_dex_sum,txt_dex_rules_number,txt_dex_evo_acuracy,txt_dex_evo_uncov,txt_dex_evo_rules_number,txt_dex_evo_rules_length,txt_dex_rules_length,txt_evo_sum,txt_evo_mean],
        outputs=[lbl_save_status, dd_saves,btn_save]
    ).then(lambda : [gr.update(visible=False)]*5, outputs=[save_panel,title_save,txt_run_name,lbl_save_status,btn_save_new])

    def wrap_overwrite(selected_save, st_data, st_layer, nb_ep, nb_lr, nb_seed, st_model, st_hist, txt_dex, txt_evo, dd_optim, dd_loss,img_struct,st_retrain,txt_dex_mean,txt_dex_sum,txt_dex_rules_number,txt_dex_evo_acuracy,txt_dex_evo_uncov,txt_dex_evo_rules_number,txt_dex_evo_rules_length,txt_dex_rules_length,txt_evo_sum,txt_evo_mean):
        if not selected_save:
            return "Error: No save selected.", gr.update()
        if(not st_retrain):
            return "Error: You changed the model architecture. Please retrain the model before overwriting the save or reload the save to get the right model version.", gr.update()
        
        return overwrite_run_logic(selected_save, st_data, st_layer, nb_ep, nb_lr, nb_seed, st_model, st_hist, {"rules": txt_dex,"rules_number": txt_dex_rules_number, "evo": txt_evo,"mean": txt_dex_mean,"sum": txt_dex_sum,"evo_uncov": txt_dex_evo_uncov,"evo_rules_number": txt_dex_evo_rules_number, "evo_acuracy": txt_dex_evo_acuracy,"evo_rules_length": txt_dex_evo_rules_length,"rules_length": txt_dex_rules_length,"evo_sum": txt_evo_sum,"evo_mean": txt_evo_mean}, dd_optim, dd_loss,img_struct)
    
    btn_save.click(
        wrap_overwrite,
        inputs=[dd_saves, st_data, st_layer, nb_ep, nb_lr, nb_seed, st_model, st_hist, txt_dex, txt_evo, dd_optim, dd_loss,img_struct,st_retrain,txt_dex_mean,txt_dex_sum,txt_dex_rules_number,txt_dex_evo_acuracy,txt_dex_evo_uncov,txt_dex_evo_rules_number,txt_dex_evo_rules_length,txt_dex_rules_length,txt_evo_sum,txt_evo_mean],
        outputs=[lbl_save_status, dd_saves,btn_save]
    ).then(lambda : [gr.update(visible=False)]*5, outputs=[save_panel,title_save,txt_run_name,lbl_save_status,btn_save_new])
   
    # LOAD EVENT
    btn_load.click(
        lambda: [gr.update(visible=True),gr.update(visible=True)],outputs=[tab_model,tab_xai]
        ).then(
        load_run_logic,
        inputs=[dd_saves],
        outputs=[
            st_data, st_model, st_datadict, st_hist, rd_src, dd_skl, fl_csv, txt_tgt, lbl_data_info, 
            plt_data, nb_ep, nb_lr, nb_seed, st_layer, plt_train, txt_dex, txt_evo,lbl_load_info,
            img_struct,btn_save,st_retrain,txt_dex_mean,txt_dex_sum,txt_dex_rules_number,txt_dex_evo_acuracy,
            txt_dex_evo_uncov,txt_dex_evo_rules_number,txt_dex_evo_rules_length,txt_dex_rules_length,txt_loss,
            txt_acc,txt_evo_sum,txt_evo_mean]
        ).then(lambda : [gr.update(visible=False)]*5, outputs=[save_panel,title_save,txt_run_name,lbl_save_status,btn_save_new])
    
    tab_xai.select(lambda: [gr.update(max_lines=10), gr.update(max_lines=10)], None, [txt_dex, txt_evo])
    def run_dex(mod, d_dict, st_data):
        if not mod: return "", ""
        feats = st_data["features"]
        txt_dex_mean = ""
        r, feats_count,feats_mean,rules_count,dexire_avg_length = get_dexire_rules(mod, d_dict, feats)
        txt_dex_sum = ""
        for feat, count in feats_count:
            if count>0:
                txt_dex_sum += f"{feat}: {count}\n"
        for feat, mean in feats_mean:
            if mean>0:
                txt_dex_mean += f"{feat}: {mean:.4f}\n"
        
        

        r_formatted = format_dexire_output(r)
        
        best, test_acc, uncov_te, eng,rules_evo_count = get_dexire_evo_rules(feats, mod, d_dict)
        count_length = 0
        for rule in best:
            count_length += (len(rule)-1)
        evo_avg_length = count_length / len(best) if len(best)>0 else 0
        evo_feats_mean,evo_feats_count = count_mean_features(best, feats)
        r_evo = format_dexire_evo_output(best, feats, eng.operator_set)

        txt_evo_sum = ""
        txt_evo_mean = ""
        for feat, count in evo_feats_count:
            if count>0:
                txt_evo_sum += f"{feat}: {count}\n"
        for feat, mean in evo_feats_mean:
            if mean>0:
                txt_evo_mean += f"{feat}: {mean:.4f}\n"
            

        return r_formatted, r_evo,gr.update(visible=True),txt_dex_sum, txt_dex_mean,rules_count,test_acc, uncov_te,rules_evo_count,evo_avg_length,dexire_avg_length,txt_evo_sum,txt_evo_mean
    btn_dexire.click(
        run_dex, 
        [st_model, st_datadict, st_data], 
        [txt_dex, txt_evo,btn_save,txt_dex_sum, txt_dex_mean,txt_dex_rules_number,txt_dex_evo_acuracy,txt_dex_evo_uncov,txt_dex_evo_rules_number,txt_dex_evo_rules_length,txt_dex_rules_length,txt_evo_sum,txt_evo_mean]
    ).then(lambda : [gr.update(visible=True)]*5, outputs=[save_panel,title_save,txt_run_name,lbl_save_status,btn_save_new])

    btn_ciu.click(run_ciu, [nb_idx, st_model, st_datadict, st_data], [plt_ciu, plt_ciu_influance])

if __name__ == "__main__":
    demo.launch(css=css)