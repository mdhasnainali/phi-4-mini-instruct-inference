#!/usr/bin/env python3
import os
import json
import time
import logging
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Phi4MiniInference:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 0, n_threads: int = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        n_threads = n_threads or max(os.cpu_count() - 1, 1)
        logger.info(f"Loading model: {model_path}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            use_mmap=True,
            use_mlock=False,
        )
        logger.info("Model loaded")

    def recommend(self, country: str, height: int, weight: int, occasion: str, time_of_day: str) -> dict:
        input_str = f"""STYLE_COUNTRY={country}
STYLE_HEIGHT={height}
STYLE_WEIGHT={weight}
STYLE_OCCASION={occasion}
STYLE_TIME={time_of_day}"""

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

        logger.info(f"Generating: {occasion} at {time_of_day}")

        start_time = time.perf_counter()
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=512,
            stop=["<|end|>", "```"]
        )
        inference_time = time.perf_counter() - start_time

        content = response["choices"][0]["message"]["content"].strip()

        try:
            result = json.loads(content)
            result["_meta"] = {"inference_time_seconds": round(inference_time, 3)}
            return result
        except json.JSONDecodeError:
            return {"recommended_dress_description": content, "_meta": {"inference_time_seconds": round(inference_time, 3)}}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phi-4-mini GGUF Outfit Recommender")
    parser.add_argument("--model", type=str, default="Phi-4-mini-instruct-Q4_K_M.gguf")
    parser.add_argument("--country", type=str, default="USA")
    parser.add_argument("--height", type=int, default=170)
    parser.add_argument("--weight", type=int, default=65)
    
    parser.add_argument("--occasion", type=str, default="Business Casual")
    parser.add_argument("--time", type=str, default="Afternoon")
    args = parser.parse_args()

    model_path = args.model if os.path.isabs(args.model) else os.path.join(os.path.dirname(__file__), "models", args.model)

    inferencer = Phi4MiniInference(model_path)
    result = inferencer.recommend(args.country, args.height, args.weight, args.occasion, args.time)

    inference_time = result.pop("_meta", {}).get("inference_time_seconds", None)
    print(json.dumps(result, indent=2))
    if inference_time:
        print(f"\nInference time: {inference_time:.3f}s")


if __name__ == "__main__":
    main()