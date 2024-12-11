# python run_benchmark.py --model_name=gpt-4o-mini --dataset_path=output.json

from weave_utils.models import LiteLLMModel, MajorityVoteModel
from weave_utils.scorers import eval_majority_vote, eval_multi_choice
import json
import weave
import asyncio
from fire import Fire

from dotenv import load_dotenv
load_dotenv()


def load_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data['eval_data']


def run_benchmark(
    model_name: str = "gpt-4o-mini",
    dataset_path: str = "simple_bench_public.json",
    num_responses: int = 1,
    entity: str = None,
    project: str = "simple_bench",
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
            Default is "simple_bench".

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

    model = LiteLLMModel(model_name=model_name)

    if num_responses > 1:
        model = MajorityVoteModel(model=model, num_responses=num_responses)

    asyncio.run(evaluation.evaluate(model))


if __name__ == "__main__":
    Fire(run_benchmark)
