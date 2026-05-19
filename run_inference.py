#!/usr/bin/env python3
import os
import json
import time
from llama_cpp import Llama

MODEL_PATH = "models/Phi-4-mini-instruct-Q4_K_M.gguf"

FIXED_INPUT = {
    "STYLE_COUNTRY": "USA",
    "STYLE_HEIGHT": 170,
    "STYLE_WEIGHT": 65,
    "STYLE_OCCASION": "Business Casual",
    "STYLE_TIME": "Afternoon"
}


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Run: uv run python download_model.py")
        return

    llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=6, use_mmap=True)

    input_str = "\n".join(f"{k}={v}" for k, v in FIXED_INPUT.items())

    user_prompt = f"""Recommend an outfit based on:

{input_str}

Output valid JSON with complete_outfits array. Each outfit must include:
- name: string (e.g., 'Business Casual Friday')
- description: overall description
- occasion_type: e.g., 'business_casual'
- season: at least one of ['Spring', 'Summer', 'Autumn', 'Winter']
- body_shape: list of suitable body shapes
- top: category, name, description, color, hex_code, fabric, fit, neckline, sleeve_length, pattern, occasion_suitability, season_suitability, body_shape_suitability, styling_tips
- bottom: category, name, description, color, hex_code, fabric, fit, waist_type, length, pattern, occasion_suitability, season_suitability, body_shape_suitability, styling_tips or null
- footwear: category, name, description, color, hex_code, material, heel_height, occasion_suitability, season_suitability, styling_tips
- outerwear: category, name, description, color, hex_code, fabric, fit, warmth_level, length, occasion_suitability, season_suitability, styling_tips or null
- accessories: array of category, name, description, color, hex_code, material, metal_type, stone, occasion_suitability, styling_tips or null
- overall_color_palette: color_name, hex_code, reason
- styling_tips: string array
- why_this_outfit: 2-3 sentence explanation"""

    messages = [
        {"role": "system", "content": "You are a fashion stylist. Output valid JSON only."},
        {"role": "user", "content": user_prompt}
    ]

    start_time = time.perf_counter()
    response = llm.create_chat_completion(messages=messages, temperature=0.1, max_tokens=512)
    inference_time = time.perf_counter() - start_time

    content = response["choices"][0]["message"]["content"].strip()
    for marker in ["```json", "```"]:
        content = content.replace(marker, "").strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"recommended_dress_description": content}

    print(json.dumps(result, indent=2))
    print(f"\nInference time: {inference_time:.3f}s ({len(content.split())} tokens)")


if __name__ == "__main__":
    main()