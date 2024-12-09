# python run_benchmark.py --model_name=gpt-4o-mini --dataset_path=output.json

from weave_utils.models import LiteLLMModel, MajorityVoteModel
from weave_utils.scorers import eval_majority_vote, eval_multi_choice
import json
import weave
import asyncio
import argparse


def load_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data['eval_data']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt_4o",
        help="Model to benchmark.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="output.json",
        help="Dataset to benchmark.",
    )
    parser.add_argument(
        "--entity", type=str, default="wandb_entity", help="Name of the W&B entity."
    )
    parser.add_argument(
        "--project",
        type=str,
        default="wandb_project",
        help="Name of the W&B project.",
    )
    parser.add_argument(
        "--num_responses",
        type=int,
        default=1,
        help="Number of responses to use for majority vote.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    weave.init(f"{args.entity}/{args.project}")

    evaluation = weave.Evaluation(
        dataset=load_dataset(args.dataset_path),
        scorers=[eval_majority_vote if args.num_responses > 1 else eval_multi_choice],
        trials=1,
    )

    model = LiteLLMModel(model_name=args.model_name)

    
    if args.num_responses > 1:
        model = MajorityVoteModel(model=model, num_responses=args.num_responses)

    asyncio.run(evaluation.evaluate(model))
