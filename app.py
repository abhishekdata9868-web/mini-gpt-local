import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)


def respond(message, history):
    history = history or []
    messages = [{"role": "system", "content": "You are Mini GPT, a helpful and concise AI assistant. Answer in the user's language when possible."}]
    for item in history:
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}:
            messages.append({"role": item["role"], "content": item.get("content", "")})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            if item[0]:
                messages.append({"role": "user", "content": item[0]})
            if item[1]:
                messages.append({"role": "assistant", "content": item[1]})
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


demo = gr.ChatInterface(
    fn=respond,
    title="Mini GPT",
    description="A local AI chatbot running an open-source model. No OpenAI/OpenRouter API key required.",
    textbox=gr.Textbox(placeholder="Message Mini GPT...", container=False, scale=7),
)

demo.launch(share=True)
