import gradio as gr
import fal_gradio

gr.load(
    name='fal-ai/ltx-video',
    src=fal_gradio.registry,
).launch()