# FairBench API Routes

## POST /api/report
Submit a benchmark report. This is the payload sent by `fairbench`'s reporter (`fairbench/reporter.py`).

**Required fields:** `model_name`, `quantization` (missing either → `400`).

**Request body (current format, as sent by the CLI):**
```json
{
  "run_id": "abc123",
  "model_name": "qwen2.5-7b-instruct",
  "quantization": "q4_k_m",
  "n_ctx": 4096,
  "n_gpu_layers": -1,
  "n_threads": 8,
  "timestamp": "2026-10-23_110000",
  "category": "all",
  "hardware": {
    "cpu_model": "Intel i7",
    "cpu_cores": 8,
    "cpu_threads": 16,
    "ram_gb": 32,
    "gpu_type": "nvidia",
    "gpu_model": "RTX 4070 Ti",
    "gpu_vram_gb": 12,
    "os": "Linux 6.8"
  },
  "quality_scores": {
    "mmlu": 0.623,
    "hellaswag": 0.834,
    "gsm8k": 0.712
  },
  "speed": {
    "ttft_ms": 145.2,
    "tpot_ms": 78.1,
    "throughput": 78.1
  },
  "memory": {
    "vram_peak_gb": 4.8,
    "cpu_ram_peak_gb": 6.2
  },
  "swebench": {
    "resolved": 3,
    "total": 10
  }
}
```

> **Note:** `speed.tpot_ms` carries a token *rate* (tokens/s) despite the column name — see README "Speed". A code fix to rename/recompute is tracked separately.

**Legacy format:** the server also accepts older clients that send nested `model: {name, quantization}` and top-level `id`; these are mapped to the same columns. Unknown/missing optional fields default to `0` / `"none"` / generated UUID (`run_id`) / current time.

**Response (`201`):**
```json
{
  "success": true,
  "report": {
    "id": "abc123",
    "timestamp": "2026-10-23_110000",
    "published": true
  }
}
```

Errors: `400` invalid body or missing required fields, `500` database error (`{"error": ...}`).

## GET /api/leaderboard
Get the leaderboard with optional filters.

**Query params:**
- `model` (optional): filter by model name (ILIKE)
- `gpu` (optional): filter by GPU model (ILIKE)
- `min_score` (optional): minimum average quality score

**Response:**
```json
{
  "reports": [
    {
      "id": "abc123",
      "timestamp": "2026-10-23_110000",
      "model_name": "qwen2.5-7b-instruct",
      "quantization": "q4_k_m",
      "cpu_model": "Intel i7",
      "cpu_cores": 8,
      "ram_gb": 32,
      "gpu_type": "nvidia",
      "gpu_model": "RTX 4070 Ti",
      "gpu_vram_gb": 12,
      "quality_mmlu": 0.623,
      "quality_hellaswag": 0.834,
      "quality_gsm8k": 0.712,
      "speed_ttft_ms": 145.2,
      "speed_tpot_ms": 12.8,
      "speed_throughput": 78.1,
      "memory_vram_peak_gb": 4.8,
      "memory_cpu_ram_peak_gb": 6.2
    }
  ]
}
```

## GET /api/models
Get model statistics (averages across all hardware).

**Response:**
```json
{
  "models": [
    {
      "name": "qwen2.5-7b-instruct",
      "avg_mmlu": 0.615,
      "avg_hellaswag": 0.828,
      "avg_gsm8k": 0.705,
      "avg_throughput": 75.3,
      "avg_vram_gb": 4.5,
      "sample_count": 42
    }
  ]
}
```
