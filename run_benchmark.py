# python run_benchmark.py --model_name=gpt-4o-mini --dataset_path=output.json

from typing import Optional
import json
import weave
import asyncio
from fire import Fire

from dotenv import load_dotenv
load_dotenv()

from weave_utils.models import LiteLLMModel, MajorityVoteModel
from weave_utils.scorers import eval_majority_vote, eval_multi_choice


def load_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data['eval_data']


def run_benchmark(
    model_name: str = "gpt-4o-mini",
    dataset_path: str = "simple_bench_public.json",
    num_responses: int = 1,
    entity: str = "simplebench",
    project: str = "simple_bench_public",
    temp: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    max_retries: int = 3,
    system_prompt_path: str = "system_prompt.txt",
):
    """
    Run a benchmark evaluation on a given model and dataset.

    Args:
        model_name (str): Name of the model to use for inference.
            Default is "gpt-4o-mini".
        dataset_path (str): Path to the dataset JSON file.
            Default is "simple_bench_public.json".
        num_responses (int): If greater than 1, majority voting will be applied.
            Default is 1 (no majority voting).
        entity (str): Optional Weave entity (org/user name) for evaluation tracking.
        project (str): The project name under the specified entity.
            Default is "simple_bench_public".
        temp (float): Temperature for the model.
            Default is 0.7.
        max_tokens (int): Maximum number of tokens to generate.
            Default is 2048.
        top_p (float): Top-p for the model.
            Default is 0.95.
        max_retries (int): Maximum number of retries for the model.
            Default is 3.
        system_prompt (str): System prompt for the model.
            Default is "You are an expert at reasoning and you always pick the most realistic answer. Think step by step and output your reasoning followed by your final answer using the following format: Final Answer: X where X is one of the letters A, B, C, D, E, or F."

    Example:
        python run_benchmark.py --model_name=gpt-4o-mini --dataset_path=simple_bench_public.json --num_responses=3
    """

    if entity is not None:
        weave.init(f"{entity}/{project}")
    else:
        weave.init(f"{project}")

    evaluation = weave.Evaluation(
        dataset=load_dataset(dataset_path),
        scorers=[eval_majority_vote if num_responses > 1 else eval_multi_choice],
        trials=1,
    )

    with open(system_prompt_path, "r") as f:
        system_prompt = f.read().strip()

    model = LiteLLMModel(
        model_name=model_name,
        temp=temp,
        max_tokens=max_tokens,
        top_p=top_p,
        max_retries=max_retries,
        system_prompt=system_prompt
    )

    if num_responses > 1:
        model = MajorityVoteModel(model=model, num_responses=num_responses)

    asyncio.run(evaluation.evaluate(model))


if __name__ == "__main__":
    Fire(run_benchmark)
