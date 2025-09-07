from pathlib import Path
from typing import Union
import pandas as pd


def generate_verb_intelligence_data(
    file_to_open: Union[str, Path],
    target_name: Union[str, Path],
    random_state: int = 47,
) -> None:
    file_to_open = Path(file_to_open)
    target_name = Path(target_name)

    data = pd.read_csv(file_to_open)

    # Sample 40 travel posts
    travel_posts = data[data["post_travel"].notnull()]["post_travel"].sample(40, random_state=random_state)
    # Sample 40 abortion posts
    abortion_posts = data[data["post_abortion"].notnull()]["post_abortion"].sample(40, random_state=random_state)
    # Sample 40 ai posts
    ai_posts = data[data["post_ai"].notnull()]["post_ai"].sample(40, random_state=random_state)

    # Concatenate the posts
    posts = pd.concat([travel_posts, abortion_posts, ai_posts], ignore_index=True)

    # Change the column name
    posts.rename("post", inplace=True)

    posts.to_csv(target_name, index=False)

if __name__ == "__main__":
    generate_verb_intelligence_data(
        file_to_open="data/split/full_test.csv",
        target_name="data/reporting/verbal_intelligence_test.csv",
        random_state=47,
    )
