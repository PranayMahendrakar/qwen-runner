from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading Qwen2.5-0.5B-Instruct... (~1GB download on first run)")

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="./model",
    torch_dtype=torch.float32,
)

device = torch.device("cpu")
model.to(device)
model.eval()

print("Qwen2.5 loaded successfully!")

# Sample prompts to run in CI (no interactive input)
prompts = [
    "What is quantum computing?",
    "Give me 3 tips for writing clean Python code.",
    "Write a haiku about artificial intelligence.",
]

for user_input in prompts:
    print(f"\n{'='*50}")
    print(f"You: {user_input}")

    messages = [
        {"role": "system", "content": "You are Qwen, a helpful AI assistant created by Alibaba Cloud."},
        {"role": "user", "content": user_input}
    ]

    # Apply Qwen's built-in chat template
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
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"Qwen: {response}")

print(f"\n{'='*50}")
print("Qwen2.5-0.5B inference complete!")
