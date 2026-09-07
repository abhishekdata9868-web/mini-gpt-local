# Mini GPT Local

A beginner-friendly ChatGPT-style chatbot that runs an open-source language model locally. **No OpenAI, OpenRouter, Gemini, or other AI API key is required.**

## Model

This project uses the public `Qwen/Qwen2.5-0.5B-Instruct` model from Hugging Face. The model is downloaded to the runtime and inference happens locally in the Colab runtime.

## Run in Google Colab

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

## Important

- No API key is used.
- A Colab GPU is recommended for better speed.
- The first run downloads the model, so it can take a few minutes.
- The public Gradio link only works while the Colab runtime is running.
- This is a learning/testing project, not permanent hosting.
