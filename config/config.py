MODEL_CONFIG = {
    "repo_id": "unsloth/Phi-4-mini-instruct-GGUF",
    "filename": "Phi-4-mini-instruct-Q4_K_M.gguf",
    "quantization": "Q4_K_M",
    "size_mb": 2490,
}

INFERENCE_CONFIG = {
    "n_ctx": 4096,
    "n_gpu_layers": 0,
    "max_tokens": 1024,
    "temperature": 0.1,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}

DEFAULT_INPUT = {
    "country": "USA",
    "height": 170,
    "weight": 65,
    "occasion": "Business Casual",
    "time": "Afternoon"
}