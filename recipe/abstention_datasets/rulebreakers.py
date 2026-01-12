"""
Adapted from recipe/abstention_datasets/squad.py
"""

import datasets
from recipe.abstention_datasets.abstract_abstention_dataset import AbstentionDataset
from recipe.abstention_datasets.abstract_abstention_dataset import Prompt
from pathlib import Path
import pandas as pd
import urllib.request

class RulebreakersDataset(AbstentionDataset):

    _PREPROMPT = ""  # nb: _preprompt is not used for rulebreakers
    _TEMPLATE = "{preprompt}Premises: {context}\n{question}"

    def __init__(self, max_num_samples=None):
        super().__init__()

        self.dataset = self.load_dataset()
        self.max_num_samples = max_num_samples

    def load_dataset(self) -> datasets.Dataset:
        repo_dir = Path(__file__).absolute().parent.parent.parent
        print(repo_dir)
        data_path = repo_dir / Path("data/rulebreakers.parquet")
        # download
        if not data_path.exists():
            self.download(data_path)
        df = pd.read_parquet(data_path)
        df_dict = df[df["category"] != "cities"].copy().to_dict("list")
        # filtering out the 'geographical' subset of the whole dataset (resulting in 1.82k rulebreakers,
        # 1.82k non-rulebreakers - therefore 1.82k x 2 = 3.64k total questions)
        dataset = datasets.Dataset.from_dict(df_dict)
        return dataset

    def download(self, data_path: Path):
        url = "https://huggingface.co/datasets/jason-c/rulebreakers/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
        try:
            urllib.request.urlretrieve(url, data_path)
        except Exception as e:
            print(f"Failed to download dataset from {url}."
                  f" Download to {data_path} manually.")
            raise e

    def __len__(self):
        return self.max_num_samples or len(self.dataset)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        item = self.dataset[idx]

        question = self._TEMPLATE.format(
            preprompt=self._PREPROMPT,
            context=f"Suppose we are told that {item['premise1'].replace('If ', 'if ')} "
                    f"As a matter of fact, {item['premise2']}",  # .replacing with lowercase 'if' for grammaticality
            question="What conclusion, if any, follows from the Premises? "
                     "If you think nothing follows from the Premises, answer 'Nothing follows'.",
        )
        should_abstain = "Nothing follows" in item["target_correct_conclusion"]
        reference_answers = (
            [item["target_correct_conclusion"]] if not should_abstain else None
        )
        metadata = {"rulebreakers_id": item["id"]}

        return Prompt(
            question=question,
            reference_answers=reference_answers,
            should_abstain=should_abstain,
            metadata=metadata,
        )


if __name__ == "__main__":
    dataset = RulebreakersDataset(max_num_samples=5)
    for i in range(len(dataset)):
        print(dataset[i])
        print()