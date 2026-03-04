from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading Qwen2.5-0.5B-Instruct...")

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="./model",
    torch_dtype=torch.float32,
)

device = torch.device("cpu")
model.to(device)
model.eval()

print("Qwen2.5 loaded!\n")

# Ask: who are you
user_input = "Who are you?"

print(f"You: {user_input}")

messages = [
    {"role": "system", "content": "You are Qwen, a helpful AI assistant created by Alibaba Cloud."},
    {"role": "user", "content": user_input}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

encoded = tokenizer([text], return_tensors="pt").to(device)
input_length = encoded["input_ids"].shape[1]

with torch.no_grad():
    outputs = model.generate(
        **encoded,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = outputs[0][input_length:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

print(f"\nQwen: {response}")
print("\n=== Done ===")
