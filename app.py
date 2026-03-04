from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os

app = Flask(__name__)
CORS(app)

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
print("Model ready!")


@app.route("/")
def home():
    return jsonify({"status": "Qwen2.5 API running"})


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data = request.json
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "No messages"}), 400
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer([text], return_tensors="pt").to(device)
        input_len = encoded["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(
            out[0][input_len:], skip_special_tokens=True
        ).strip()
        return jsonify({"response": resp})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
