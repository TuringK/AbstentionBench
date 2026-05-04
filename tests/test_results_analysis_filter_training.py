# Training-data overlap filtering for benchmark results.


import json

import pandas as pd
import pytest

from analysis.results_analysis import filter_training_data


def test_legacy_csv_training_removes_matching_prompts(tmp_path):
    """CSV paired datasets remain supported; breaking this silently biases sweeps against legacy runs."""
    df = pd.DataFrame(
        {
            "prompt_question": ["What is 2 + 2?", "Hello world"],
            "dataset_name": ["Foo", "Foo"],
        }
    )
    train_path = tmp_path / "train.csv"
    pd.DataFrame({"question": ["What is 2 + 2?"]}).to_csv(train_path, index=False)

    filtered = filter_training_data(df.copy(), str(train_path))
    assert list(filtered["prompt_question"]) == ["Hello world"]


def test_json_training_removes_matching_prompts(tmp_path):
    """JSON training files match the extractor-facing schema (list of dicts with ``question``)."""
    df = pd.DataFrame(
        {
            "prompt_question": ["Q1", "Q2"],
            "dataset_name": ["Foo", "Foo"],
        }
    )
    train_path = tmp_path / "train.json"
    train_path.write_text(
        json.dumps(
            [
                {
                    "dataset": "X",
                    "question": "Q1",
                    "positive": "...",
                    "negative": "...",
                    "task": "underspecified context",
                    "should_abstain": True,
                }
            ]
        )
    )

    filtered = filter_training_data(df.copy(), str(train_path))
    assert list(filtered["prompt_question"]) == ["Q2"]


def test_raises_when_results_has_no_prompt_column(tmp_path):
    """Without a recognizable prompt column we cannot compare to training; mis-detection must fail loudly."""
    df = pd.DataFrame({"not_a_question": ["x"]})
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps([{"question": "x"}]))

    with pytest.raises(ValueError, match="question column"):
        filter_training_data(df, str(train_path))


def test_whitespace_only_difference_still_matches_training_prompt(tmp_path):
    """Normalization collapses internal whitespace; train/eval prompts often differ only by spacing."""
    df = pd.DataFrame({"prompt_question": ["What  is  2 + 2?", "Hello world"]})
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps([{"question": "What is 2 + 2?"}]))

    filtered = filter_training_data(df.copy(), str(train_path))
    assert list(filtered["prompt_question"]) == ["Hello world"]


def test_json_objects_without_question_key_are_ignored(tmp_path):
    """Rows without ``question`` must not crash the loader and must not add phantom training prompts."""
    df = pd.DataFrame({"prompt_question": ["Q1", "Q2"]})
    train_path = tmp_path / "train.json"
    train_path.write_text(
        json.dumps(
            [
                {
                    "positive": "...",
                    "negative": "...",
                    "task": "t",
                    "should_abstain": True,
                }
            ]
        )
    )

    filtered = filter_training_data(df.copy(), str(train_path))
    assert list(filtered["prompt_question"]) == ["Q1", "Q2"]


def test_raises_when_training_json_root_is_not_a_list(tmp_path):
    """A non-list JSON root is always wrong for this pipeline; loading must error instead of guessing."""
    df = pd.DataFrame({"prompt_question": ["x"]})
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps({"question": "x"}))

    with pytest.raises(ValueError, match="list of examples"):
        filter_training_data(df, str(train_path))


@pytest.mark.parametrize(
    "question_col",
    ["prompt_question", "question"],
    ids=["primary_prompt_question", "fallback_question_column"],
)
def test_filter_works_for_either_standard_results_prompt_column(tmp_path, question_col):
    """Primary tables use `prompt_question`. Other exports may only have `question`. Both must match training."""
    df = pd.DataFrame({question_col: ["Match me", "Leave me"]})
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps([{"question": "Match me"}]))

    filtered = filter_training_data(df.copy(), str(train_path))
    assert list(filtered[question_col]) == ["Leave me"]


def test_jsonl_extension_uses_same_loader_as_json(tmp_path):
    """`.jsonl` is an accepted alias (see extractors). Behaviour must match `.json` for a JSON array file."""
    df = pd.DataFrame({"prompt_question": ["A", "B"]})
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(json.dumps([{"question": "A"}]))

    filtered = filter_training_data(df.copy(), str(train_path))
    assert list(filtered["prompt_question"]) == ["B"]


def test_null_prompt_never_matches_literal_none_in_training(tmp_path):
    """Missing prompts used to stringify to `"None"`, which could spuriously match a real training question."""
    df = pd.DataFrame({"prompt_question": [None, "Keep me"]}, dtype=object)
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps([{"question": "None"}]))

    filtered = filter_training_data(df.copy(), str(train_path))
    assert filtered["prompt_question"].tolist() == [None, "Keep me"]
