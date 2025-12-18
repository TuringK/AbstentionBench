"""
CSV-based Abstention Dataset
============================
Loads evaluation data from a local CSV file.

Add to recipe/abstention_datasets/__init__.py (if it exists) or just import directly.
"""

import ast
import pandas as pd
from typing import Optional

from recipe.abstention_datasets.abstract_abstention_dataset import (
    AbstentionDataset,
    Prompt,
)


class CSVAbstentionDataset(AbstentionDataset):
    """
    Load abstention evaluation data from a CSV file.
    
    Expected CSV columns:
        - question: str (required)
        - should_abstain: bool (required)
        - reference_answers: str (optional, can be stringified list)
        - Other columns become metadata
    """
    
    def __init__(
        self,
        csv_path: str,
        max_num_samples: Optional[int] = None,
    ):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        
        if max_num_samples is not None:
            self.df = self.df.head(max_num_samples)
        
        # Validate required columns
        required_cols = ["question", "should_abstain"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"CSV missing required column: {col}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx) -> Prompt:
        if idx >= len(self):
            raise IndexError
        
        row = self.df.iloc[idx]
        
        # Parse reference_answers if present
        reference_answers = None
        if "reference_answers" in row and pd.notna(row["reference_answers"]):
            ref_ans = row["reference_answers"]
            # Handle if it's a stringified list like "['answer1', 'answer2']"
            if isinstance(ref_ans, str):
                try:
                    reference_answers = ast.literal_eval(ref_ans)
                except (ValueError, SyntaxError):
                    # It's just a plain string
                    reference_answers = [ref_ans]
            elif isinstance(ref_ans, list):
                reference_answers = ref_ans
        
        # Build metadata from other columns
        metadata_cols = [
            col for col in self.df.columns 
            if col not in ["question", "should_abstain", "reference_answers"]
        ]
        metadata = {}
        for col in metadata_cols:
            val = row[col]
            if pd.notna(val):
                # Convert numpy types to native Python types for JSON serialization
                if hasattr(val, 'item'):  # numpy scalar
                    val = val.item()
                metadata[col] = val
        
        # Ensure should_abstain is a native Python bool
        should_abstain_val = row["should_abstain"]
        if hasattr(should_abstain_val, 'item'):  # numpy scalar
            should_abstain_val = should_abstain_val.item()
        
        return Prompt(
            question=row["question"],
            reference_answers=reference_answers,
            should_abstain=bool(should_abstain_val),
            metadata=metadata,
        )


class BalancedEvalDataset(CSVAbstentionDataset):
    """
    Evaluation dataset from balanced_dataset_with_annotations.csv
    """
    
    def __init__(self, max_num_samples: Optional[int] = None):
        csv_path = "/mnt/parscratch/users/acb20df/private/TuringProj/AbstentionBench/PromptTuning/data/balanced_dataset_with_annotations.csv"
        super().__init__(csv_path=csv_path, max_num_samples=max_num_samples)