from transformers import AutoTokenizer, AutoModelForCausalLM

# Model path from Hugging Face
MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"

# Download the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_safetensors=True)

# Save to a local directory
tokenizer.save_pretrained("./phi3_mini_local/")
model.save_pretrained("./phi3_mini_local/")
