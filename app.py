from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os, threading

app = Flask(__name__)
CORS(app)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
print("Loading Qwen2.5-0.5B-Instruct...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat1,
    low_cpu_mem_usage=True,
)
model.eval()

# Pre-warm the model with a dummy inference so first real request is fast
def warmup():
    try:
        warm_msgs = [{"role":"user","content":"hi"}]
        text = tokenizer.apply_chat_template(warm_msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer([text], return_tensors="pt")
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        print("Model warmed up!")
    except Exception as e:
        print(f"Warmup error: {e}")

threading.Thread(target=warmup, daemon=True).start()
print("Model ready!")

# Request lock - queue requests so they don't overlap
lock = threading.Lock()

@app.route("/")
def home():
    return jsonify({"status": "online", "model": "Qwen2.5-0.5B-Instruct"})

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data = request.json
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "No messages"}), 400

    # Only keep last 6 messages to keep context short and fast
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]
    trimmed = sys_msgs + other_msgs[-6:]

    with lock:
        try:
            text = tokenizer.apply_chat_template(
                trimmed,
                tokenize=False,
                add_generation_prompt=True
            )
            encoded = tokenizer([text], return_tensors="pt")
            input_len = encoded["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **encoded,
                    max_new_tokens=150,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            resp = tokenizer.decode(
                out[0][input_len:],
                skip_special_tokens=True
            ).strip()
            return jsonify({"response": resp})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
