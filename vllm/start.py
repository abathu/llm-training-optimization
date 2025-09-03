from vllm import LLM, SamplingParams

# Choose model (from HF Hub or local cache)
model_name = "Qwen/Qwen2.5-7B-Instruct"

# Initialize vLLM engine
llm = LLM(
    model=model_name,
    tensor_parallel_size=2,   # you have 2x A6000
    dtype="bfloat16"          # match your training setting
)

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=128
)

# Example prompt
prompt = "Explain the difference between civil law and criminal law in simple terms."

# Run inference
outputs = llm.generate([prompt], sampling_params)

# Print results
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
llm.shutdown()