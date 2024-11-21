"""
TODO:
- component: grid with pics (icons or text or empty)
- function: update grid as directional input is received
- algorithms: use logic from alg to figure out who goes where from one board setup to the next
- state: store current board, then next
- transition: smooth the transition once next board updated, moving from prev to next

"""
import gradio as gr

def create():
    with gr.Blocks() as demo:
        textbox_comp = gr.Textbox('hi')
        
    return demo