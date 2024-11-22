# `fal-gradio`

is a Python package that makes it very easy for developers to create machine learning apps that are powered by fal's API.

# Installation

You can install `fal-gradio` directly using pip:

```bash
pip install fal-gradio
```

That's it! 

# Basic Usage

Just like if you were to use the `fal` API, you should first save your fal API key to this environment variable:

```bash
export FAL_KEY=<your token>
```

Then in a Python file, write:

```python
import gradio as gr
import fal_gradio

gr.load(
    name='fal-ai/flux',  # or any other supported fal model
    src=fal_gradio.registry,
).launch()
```

Run the Python file, and you should see a Gradio Interface connected to the model on fal!


# Customization 

Once you can create a Gradio UI from an fal endpoint, you can customize it by setting your own input and output components, or any other arguments to `gr.Interface`. For example, the screenshot below was generated with:

```py
import gradio as gr
import fal_gradio

gr.load(
    name='fal-ai/flux',
    src=fal_gradio.registry,
    title='fal-Gradio Integration',
    description="fal-ai/flux model.",
).launch()
```

# Composition

Or use your loaded Interface within larger Gradio Web UIs, e.g.

```python
import gradio as gr
import fal_gradio

with gr.Blocks() as demo:
    with gr.Tab("fal-ai/flux"):
        gr.load('fal-ai/flux', src=fal_gradio.registry)
    with gr.Tab("fal-ai/ltx-video"):
        gr.load('fal-ai/ltx-video', src=fal_gradio.registry)

demo.launch()
```

# Under the Hood

The `fal-gradio` Python library has two dependencies: `fal` and `gradio`. It defines a "registry" function `fal_gradio.registry`, which takes in a model name and returns a Gradio app.

# Supported Models in fal

The following model types are currently supported:

- Text-to-Image models:
  - fal-ai/flux
  - fal-ai/flux-dev
  - Supported parameters:
    - Prompt
    - Negative Prompt (optional)
    - Steps (1-100, default: 30)
    - Guidance Scale (1-20, default: 7.5)
    - Seed (optional)

- Text-to-Video models:
  - fal-ai/ltx-video
  - Supported parameters:
    - Prompt (multi-line input)
    - Negative Prompt (optional)
    - Steps (1-100, default: 30)
    - Guidance Scale (1-20, default: 3)
    - Seed (optional)

-------

Note: if you are getting an authentication error, then the fal API Client is not able to get the API token from the environment variable. You can set it in your Python session:

```python
import os

os.environ["FAL_KEY"] = "your-api-key-here"
```