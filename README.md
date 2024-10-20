# Simple Bench

https://simple-bench.com/

## Run Instructions

Run benchmark:
```
python run_benchmark.py --model_name=gpt-4o --dataset_path=simple_bench_public.json
```

## Setup Instructions

Clone the github repo and cd into it.

Make sure you have the correct python version (3.10.11) as a venv:
```
pyenv local 3.10.11
python -m venv llm_env
source llm_env/bin/activate
```

Install dependencies:
``` 
pip install -r requirements.txt
```

Create a `.env` file with the following:
```
OPENAI_API_KEY=<your key>
ANTHROPIC_API_KEY=<your key>
...
```
