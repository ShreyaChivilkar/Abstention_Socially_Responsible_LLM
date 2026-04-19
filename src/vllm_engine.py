from vllm import LLM, SamplingParams

class VLLMEngine:
    def __init__(self, model_name, lora_adapter=None):
        self.lora_adapter = lora_adapter
        self.lora_request = None
        self.llm = LLM(model=model_name, enable_lora=lora_adapter is not None)

        if lora_adapter is not None:
            from vllm.lora.request import LoRARequest
            self.lora_request = LoRARequest("sft_adapter", 1, lora_adapter)

    def generate(self, prompt):
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=5,
            logprobs=5
        )

        outputs = self.llm.generate([prompt], sampling_params, lora_request=self.lora_request)

        output = outputs[0]

        text = output.outputs[0].text
        logprobs = output.outputs[0].logprobs

        return text, logprobs
