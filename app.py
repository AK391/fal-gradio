import gradio as gr
import fal_gradio

gr.load(
    name='fal-ai/ltx-video/image-to-video',
    src=fal_gradio.registry,
).launch()