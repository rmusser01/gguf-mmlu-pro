# gguf-mmlu-pro
GGUF based MMLU-Pro benchmark tool

## Installation

1. The benchmark currently depends on Llamafile.
2. Grab the MMLU Pro dataset from https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro and link or copy the validation-... parquet file into the root of this repository.
3. Install the required python packages by running `pip install -r requirements.txt`.

## Running

Prepare a gguf quant file, then run:

```
python3 bench.py llama3 --llm=/llm/meta-llama_Meta-Llama-3-70B-Instruct/ggml-model-q4_k_m.gguf --debug --verbose
```

where `llama3` is the model format (supports `llama3` and `mistral` at the moment), and `--llm=` points to the gguf quant file. Remove `--debug` and/or `--verbose` if things are too noisy.

## Details
The system caches all LLM responses always, in the sample-outputs directory; even things like errors or rejections. It then always passes through these on future runs, but due to the caching, the LLM is not started unless an uncached response is requested. The LLM itself is hashed into the cache key using a "quick hash" which takes 1 MB per GB of the model file into a sha256 hash digest. Under normal circumstances, the code should realize that a model is different and not reuse the cache even if the model has the same name as a previous model that was benchmarked.
