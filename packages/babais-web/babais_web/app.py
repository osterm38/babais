import gradio as gr

def create():
    # return ''
    with gr.Blocks() as demo:
        textbox_comp = gr.Textbox('hi')
        
    return demo