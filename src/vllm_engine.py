from vllm import LLM, SamplingParams

class VLLMEngine:
    def __init__(self, model_name):
        self.llm = LLM(model=model_name)

    def generate(self, prompt):
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=5,
            logprobs=5
        )

        outputs = self.llm.generate([prompt], sampling_params)

        output = outputs[0]

        text = output.outputs[0].text
        logprobs = output.outputs[0].logprobs

        return text, logprobs