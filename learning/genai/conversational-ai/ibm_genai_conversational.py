from flask import Flask, request, render_template
from flask_cors import CORS
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

# Choose a CPU-friendly CausalLM model (No login or tokens required!)
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialized with a system prompt to give the model context
conversation_history = [
    {"role": "system", "content": "You are a helpful, light-hearted AI assistant."}
]

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    global conversation_history

    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']

    # 1. Append the new user input to the chat history
    conversation_history.append({"role": "user", "content": input_text})

    # 2. Prevent RAM crashes: Keep system prompt + last 6 messages
    if len(conversation_history) > 7:
        conversation_history = [conversation_history[0]] + conversation_history[-6:]

    # 3. Use apply_chat_template and explicitly extract raw input IDs
    chat_tokens = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True  # Force it to return a dictionary structure
    )

    # Safely extract the pure list of integers from the 'input_ids' key
    input_ids_list = chat_tokens["input_ids"]

    # Convert the plain integer list into a PyTorch Tensor with a batch dimension
    tokenized_chat = torch.tensor([input_ids_list])

    # 4. Generate response safely
    with torch.no_grad():
        outputs = model.generate(
            tokenized_chat,
            max_new_tokens=150,        # Limits generation so your CPU stays fast
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # 5. Extract and decode ONLY the newly generated tokens
    input_length = tokenized_chat.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

    # 6. Record the assistant's answer for the next turn
    conversation_history.append({"role": "assistant", "content": response})

    return response

if __name__ == '__main__':
    app.run()
