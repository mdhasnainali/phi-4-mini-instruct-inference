# Phi-4-mini GGUF Outfit Recommendation

Outfit recommendation inference using unsloth/Phi-4-mini-instruct-GGUF model with llama-cpp-python.

## Setup

```bash
# Install uv if not already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Download Model

```bash
uv run python download_model.py --filename Phi-4-mini-instruct-Q4_K_M.gguf
```

Available quantizations:
- `Q2_K.gguf` (1.68 GB)
- `Q3_K_M.gguf` (2.12 GB)
- `Q4_K_M.gguf` (2.49 GB) - default
- `Q5_K_M.gguf` (2.85 GB)
- `Q6_K.gguf` (3.16 GB)
- `Q8_0.gguf` (4.08 GB)

## Usage

### Run with fixed input values
```bash
uv run python run_inference.py
```

### Run with custom parameters
```bash
uv run python inference.py \
    --country USA \
    --height 170 \
    --weight 65 \
    --occasion "Business Casual" \
    --time "Afternoon"
```

### GPU acceleration (Linux)
```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv sync
uv run python inference.py --n_gpu_layers 32
```

## Input Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| STYLE_COUNTRY | Country for style preferences | USA |
| STYLE_HEIGHT | Height in cm | 170 |
| STYLE_WEIGHT | Weight in kg | 65 |
| STYLE_OCCASION | Occasion type | Business Casual |
| STYLE_TIME | Time of day | Afternoon |

## Output Schema

```json
{
  "complete_outfits": [
    {
      "name": "string",
      "description": "string",
      "occasion_type": "string",
      "season": ["Spring", "Summer", "Autumn", "Winter"],
      "body_shape": ["Hourglass", "Apple", "Pear", "Rectangle", "Inverted Triangle"],
      "top": {
        "category": "tshirt | shirt | blouse | ...",
        "name": "string",
        "description": "string",
        "color": "string",
        "hex_code": "#RRGGBB",
        "fabric": "string",
        "fit": "slim | regular | relaxed | oversized | tailored",
        "neckline": "string",
        "sleeve_length": "short | 3/4 | long | extra_long",
        "pattern": "string or null",
        "occasion_suitability": ["..."],
        "season_suitability": ["..."],
        "body_shape_suitability": ["..."],
        "styling_tips": ["..."]
      },
      "bottom": { ... } or null,
      "footwear": { ... },
      "outerwear": { ... } or null,
      "accessories": [{ ... }] or null,
      "overall_color_palette": [{ "color_name": "...", "hex_code": "...", "reason": "..." }],
      "styling_tips": ["..."],
      "why_this_outfit": "string"
    }
  ]
}
```

## Files

- `inference.py` - Full-featured CLI inference script
- `run_inference.py` - Simple script with fixed values
- `download_model.py` - Model downloader from HuggingFace
- `config/config.py` - Configuration settings
- `prompts/input.txt` - Input parameters and output schema
- `pyproject.toml` - Project dependencies (uv)