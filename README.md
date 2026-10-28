# Mini GPT Local

A beginner-friendly ChatGPT-style chatbot that runs an open-source language model locally. **No OpenAI, OpenRouter, Gemini, or other AI API key is required.**

## Chat app model

The chat app uses the public `Qwen/Qwen2.5-0.5B-Instruct` model from Hugging Face. The model is downloaded to the runtime and inference happens locally in the Colab runtime.

## Run the chat app in Google Colab

1. Open Google Colab: https://colab.research.google.com/
2. Create a new notebook.
3. Run:

```python
!git clone https://github.com/abhishekdata9868-web/mini-gpt-local.git
%cd mini-gpt-local
!pip install -q -r requirements.txt
!python app.py
```

4. Wait for the model to download.
5. Gradio will print a public `gradio.live` link. Open it to chat with Mini GPT.

## Train MiniGPT from scratch

This is a separate learning path from the Qwen chat app above. The `from_scratch` model starts with randomly initialized weights and learns character-level text patterns from the public Tiny Shakespeare corpus. It is not an instruction-tuned assistant and will not behave like ChatGPT.

### Google Colab

Open the [from-scratch Colab notebook](https://colab.research.google.com/github/abhishekdata9868-web/mini-gpt-local/blob/main/from_scratch/MiniGPT_From_Scratch_Colab.ipynb). It clones the repository when needed, installs the small set of dependencies, and starts training.

### Run locally

From the repository root, run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r from_scratch/requirements.txt
python from_scratch/train.py
```

The training script selects NVIDIA CUDA when available, then Apple Silicon MPS, and otherwise CPU. CPU training may be slow. The checkpoint is saved to `from_scratch/checkpoint/minigpt.pt`.

## Important

- The chat app does not require an API key.
- A Colab GPU is recommended for better chat inference and faster from-scratch training.
- The first run downloads the chat model or training corpus, so it can take a few minutes.
- The public Gradio link only works while the Colab runtime is running.
- This is a learning/testing project, not permanent hosting.
