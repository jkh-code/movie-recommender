import ast

import numpy as np
import pandas as pd


def clean_meta_data(df: pd.DataFrame, in_place: bool = False) -> pd.DataFrame:
    def stringify(obj: str, name: str = "name", sep: str = ", ") -> str:
        dicts = ast.literal_eval(obj)
        if type(dicts) is list:
            return sep.join(d[name] for d in ast.literal_eval(obj) if type(d) is dict)
        else:
            return np.NaN

    if not in_place:
        new_df = df.copy()
    else:
        new_df = df

    # Belongs to Collection
    new_df["belongs_to_collection"] = new_df["belongs_to_collection"].map(
        lambda x:
        ast.literal_eval(x).get("name", np.NaN) if x is not np.NaN and type(ast.literal_eval(x)) is dict else np.NaN
    )

    # Genres
    new_df["genres"] = new_df["genres"].map(lambda x: stringify(obj=x))

    # Production Companies
    new_df["production_companies"] = new_df["production_companies"].map(lambda x: stringify(obj=x) if x is not np.NaN else np.NaN)

    # Production Countries
    new_df["production_countries"] = new_df["production_countries"].map(lambda x: stringify(obj=x) if x is not np.NaN else np.NaN)

    # Spoken Languages
    new_df["spoken_languages"] = new_df["spoken_languages"].map(lambda x: stringify(obj=x) if x is not np.NaN else np.NaN)

    return new_df
