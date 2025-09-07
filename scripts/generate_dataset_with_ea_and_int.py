from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import os
import re


def generate_dataset_with_ea_and_int(
    int_path: Union[str, Path],
    ea_path: Union[str, Path],
    base_path: Union[str, Path],
    target_name: Union[str, Path],
) -> None:
    
    mapping_pos = {
        "Prompt violating policy": np.nan,
        "Extremely high intensity of positive emotional affect": 5,
        "High intensity of positive emotional affect": 4,
        "Average intensity of positive emotional affect": 3,
        "Low intensity of positive emotional affect": 2,
        "Extremely low intensity of positive emotional affect": 1
    }
    mapping_neg = {
        "Prompt violating policy": np.nan,
        "Extremely high intensity of negative emotional affect": 5,
        "High intensity of negative emotional affect": 4,
        "Average intensity of negative emotional affect": 3,
        "Low intensity of negative emotional affect": 2,
        "Extremely low intensity of negative emotional affect": 1
    }

    mapping_int = {
        "Prompt violating policy": np.nan,
        "Extremely high intelligence": 5,
        "High intelligence": 4,
        "Average intelligence": 3,
        "Low intelligence": 2,
        "Extremely low intelligence": 1
    }

    int_path = Path(int_path)
    ea_path = Path(ea_path)
    base_path = Path(base_path)

    files_int = os.listdir(int_path)
    files_ea = os.listdir(ea_path)
    files = []
    for file in files_int:
        files.append(int_path / file)
    for file in files_ea:
        files.append(ea_path / file)

    files

    base_data = pd.read_csv(base_path)

    data = pd.DataFrame(columns=[
        "intelligence_score",
        "positive_emotional_affect_score",
        "negative_emotional_affect_score",
        "post_type",
        "model",
        "role"
    ])

    dfs = []

    for file in files:
        df = pd.read_csv(file)
        m = re.search(r"([A-Za-z0-9\_]+).csv", str(file))
        match = re.findall(r"[A-Za-z0-9]+", m.group(0))
        if match[0] == "int":
            df["intelligence_score"] = df["score"].map(mapping_int)
        elif match[0] == "pea":
            df["positive_emotional_affect_score"] = df["score"].map(mapping_pos)
        elif match[0] == "nea":
            df["negative_emotional_affect_score"] = df["score"].map(mapping_neg)
        
        df["post_type"] = match[1]
        df["model"] = match[2]
        df["role"] = match[3]

        df = df.merge(base_data, how="left", left_on=["post"], right_on=["post_"+match[1]])

        df.drop(columns=["score",
                         "reason", 
                         "post_ai", 
                         "post_travel", 
                         "post_abortion"], inplace=True)

        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = data.groupby(
        ["post", "post_type", "model", "role"], as_index=False
        ).first()

    data.to_csv(target_name, index=False)

    print(f"Dataset shape: {data.shape}")
    print(f"Dataset saved to {target_name}")


if __name__ == "__main__":
    generate_dataset_with_ea_and_int(
        int_path="data/reporting/intelligence",
        ea_path="data/reporting/emotional_affect",
        base_path="data/processed/processed_without_ai.csv",
        target_name="data/processed/final_results.csv",
    )
