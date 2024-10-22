# app/models/custom_phi3_llm.py
from langchain.llms.base import LLM
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CustomPhi3MiniSLM(LLM):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids']

        # Generate text with streaming
        with torch.no_grad():
            output_tokens = self.model.generate(
                input_ids, 
                max_new_tokens=150, 
                do_sample=True, 
                num_return_sequences=1
            )
            response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        return response
    
    def _stream_call(self, prompt: str, stop: Optional[List[str]] = None):
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids']
        
        # Generate tokens incrementally (streaming response)
        response = ""
        with torch.no_grad():
            for output in self.model.generate(
                input_ids, 
                max_new_tokens=150, 
                do_sample=True, 
                output_scores=True,
                return_dict_in_generate=True,
                max_time=2,  # Adjust time or tokens as per the model size
            ).sequences:
                response += self.tokenizer.decode(output, skip_special_tokens=True)
                yield response  # Streaming the output token by token

    def _identifying_params(self):
        return {"model_name": "phi3-mini"}

    @property
    def _llm_type(self):
        return "custom_phi3mini"
