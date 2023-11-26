from fastai.vision.all import *
import gradio as gr

path = Path()
learn_inf = load_learner(path/'export.pkl', cpu=True)
learn = load_learner('export.pkl')

demo = gr.Interface(fn=predict, inputs=gr.Image(), outputs=gr.Label(num_top_classes=3))
demo.launch()