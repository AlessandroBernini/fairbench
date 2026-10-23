# Contributing to FairBench

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/fairbench.git`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest`

## Adding a New Model

Edit `fairbench/registry.py` and add your model to `MODEL_REGISTRY`:

```python
"your-model-name": {
    "repo": "huggingface-user/model-repo",
    "file_pattern": "model-q4_k_m.gguf",
    "size_b": 7.0,
    "context": 32768,
},
```

## Code Style

- Use `ruff` for linting: `ruff check .`
- Use `mypy` for type checking: `mypy fairbench/`
- Follow PEP 8

## Submitting Results

After running a benchmark, your results are saved to `~/.fairbench/results/`.
To publish: `fairbench publish`

## License

By contributing, you agree that your contributions are licensed under Apache 2.0.
