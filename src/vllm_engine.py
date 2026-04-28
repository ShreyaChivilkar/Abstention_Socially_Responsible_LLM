from vllm import LLM, SamplingParams

class VLLMEngine:
    def __init__(self, model_name, lora_adapter=None, gpu_memory_utilization=None, max_model_len=None, generation_max_tokens=5):
        self.lora_adapter = lora_adapter
        self.lora_request = None
        self.generation_max_tokens = generation_max_tokens
        llm_kwargs = {
            "model": model_name,
            "enable_lora": lora_adapter is not None,
        }
        if gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**llm_kwargs)

        if lora_adapter is not None:
            from vllm.lora.request import LoRARequest
            self.lora_request = LoRARequest("sft_adapter", 1, lora_adapter)

    def generate(self, prompt):
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.generation_max_tokens,
            logprobs=5
        )

        outputs = self.llm.generate([prompt], sampling_params, lora_request=self.lora_request)

        output = outputs[0]

        text = output.outputs[0].text
        logprobs = output.outputs[0].logprobs

        return text, logprobs
