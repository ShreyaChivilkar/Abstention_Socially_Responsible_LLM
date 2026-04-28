from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TransformerEngine:
    def __init__(self, model_name, lora_adapter=None, generation_max_tokens=5):
        self.generation_max_tokens = generation_max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto"
        )

        if lora_adapter is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_adapter)

        self.model.eval()

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.generation_max_tokens,
            do_sample=False
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
