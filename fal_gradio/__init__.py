import os
import fal_client
import gradio as gr
from typing import Callable, Dict, Any, List, Tuple
from PIL import Image
import io
import base64
import numpy as np

__version__ = "0.0.1"

# Define MODEL_TO_PIPELINE first
MODEL_TO_PIPELINE = {
    "fal-ai/flux": "text-to-image",
    "fal-ai/flux-dev": "text-to-image",
    "fal-ai/ltx-video": "text-to-video",
}

# Then update it with new mappings
MODEL_TO_PIPELINE.update({
    "fal-ai/ltx-video": "text-to-video",
})

# Add to PIPELINE_REGISTRY
PIPELINE_REGISTRY = {
    "text-to-image": {
        "inputs": [
            ("prompt", gr.Textbox, {"label": "Prompt"}),
            ("negative_prompt", gr.Textbox, {"label": "Negative Prompt", "optional": True}),
            ("num_inference_steps", gr.Slider, {"label": "Steps", "minimum": 1, "maximum": 100, "value": 30, "optional": True}),
            ("guidance_scale", gr.Slider, {"label": "Guidance Scale", "minimum": 1, "maximum": 20, "value": 7.5, "optional": True}),
            ("seed", gr.Number, {"label": "Seed", "optional": True})
        ],
        "outputs": [("images", gr.Gallery, {})],
        "preprocess": lambda *args: {
            "arguments": {
                k: v for k, v in zip([
                    "prompt", "negative_prompt", "num_inference_steps",
                    "guidance_scale", "seed"
                ], args) if v is not None and v != ""
            }
        },
        "postprocess": lambda x: x["images"] if isinstance(x, dict) and "images" in x else x
    },

    "text-to-video": {
        "inputs": [
            ("prompt", gr.Textbox, {"label": "Prompt", "lines": 5}),
            ("negative_prompt", gr.Textbox, {"label": "Negative Prompt", "optional": True}),
            ("num_inference_steps", gr.Slider, {"label": "Steps", "minimum": 1, "maximum": 100, "value": 30, "optional": True}),
            ("guidance_scale", gr.Slider, {"label": "Guidance Scale", "minimum": 1, "maximum": 20, "value": 3, "optional": True}),
            ("seed", gr.Number, {"label": "Seed", "optional": True})
        ],
        "outputs": [("video", gr.Video, {"label": "Generated Video"})],
        "preprocess": lambda *args: {
            "arguments": {
                k: v for k, v in zip([
                    "prompt", "negative_prompt", "num_inference_steps",
                    "guidance_scale", "seed"
                ], args) if v is not None and v != ""
            }
        },
        "postprocess": lambda x: x["video"]["url"] if isinstance(x, dict) and "video" in x else x
    }
}

def create_component(comp_type: type, name: str, config: Dict[str, Any]) -> gr.components.Component:
    config = config.copy()
    is_optional = config.pop('optional', False)
    
    if is_optional:
        label = config.get('label', name)
        config['label'] = f"{label} (Optional)"
    
    return comp_type(label=config.get("label", name), **{k:v for k,v in config.items() if k != "label"})

def get_pipeline(model: str) -> str:
    return MODEL_TO_PIPELINE.get(model)

def get_interface_args(pipeline: str) -> Tuple[List, List, Callable, Callable]:
    if pipeline not in PIPELINE_REGISTRY:
        raise ValueError(f"Unsupported pipeline: {pipeline}")
    
    config = PIPELINE_REGISTRY[pipeline]
    
    inputs = [create_component(comp_type, name, conf) 
             for name, comp_type, conf in config["inputs"]]
    
    outputs = [create_component(comp_type, name, conf) 
              for name, comp_type, conf in config["outputs"]]
    
    return inputs, outputs, config["preprocess"], config["postprocess"]

def get_fn(model_name: str, preprocess: Callable, postprocess: Callable, api_key: str):
    def fn(*args):
        inputs = preprocess(*args)
        try:
            result = fal_client.subscribe(
                model_name,
                **inputs
            )
            return postprocess(result)
        except Exception as e:
            raise gr.Error(f"Model prediction failed: {str(e)}")
    return fn

def get_image_base64(url: str, ext: str):
    with open(url, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return "data:image/" + ext + ";base64," + encoded_string

def handle_user_msg(message: str):
    if type(message) is str:
        return message
    elif type(message) is dict:
        if message["files"] is not None and len(message["files"]) > 0:
            ext = os.path.splitext(message["files"][-1])[1].strip(".")
            if ext.lower() in ["png", "jpg", "jpeg", "gif", "pdf"]:
                encoded_str = get_image_base64(message["files"][-1], ext)
            else:
                raise NotImplementedError(f"Not supported file type {ext}")
            content = [
                    {"type": "text", "text": message["text"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encoded_str,
                        }
                    },
                ]
        else:
            content = message["text"]
        return content
    else:
        raise NotImplementedError

def registry(name: str | Dict, token: str | None = None, inputs=None, outputs=None, **kwargs):
    """
    Create a Gradio Interface for a model on fal.
    
    Parameters:
        - name (str | Dict): The name of the fal model
        - token (str, optional): The API token for fal
        - inputs (List[gr.Component], optional): Custom input components
        - outputs (List[gr.Component], optional): Custom output components
    """
    if isinstance(name, dict):
        model_name = name.get('name', name.get('model_name', ''))
    else:
        model_name = name

    api_key = token or os.environ.get("FAL_KEY")
    if not api_key:
        raise ValueError("FAL_KEY environment variable is not set.")
    
    pipeline = get_pipeline(model_name)
    inputs_, outputs_, preprocess, postprocess = get_interface_args(pipeline)
    
    inputs = inputs or inputs_
    outputs = outputs or outputs_

    fn = get_fn(model_name, preprocess, postprocess, api_key)
    return gr.Interface(fn=fn, inputs=inputs, outputs=outputs, **kwargs)