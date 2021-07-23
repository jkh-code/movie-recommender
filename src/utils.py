import ast
import numpy as np
import pandas as pd
import re


def stringify(obj: str, name: str = "name", sep: str = ", ") -> str:
    """
    This function takes a string representations of a list of dictionaries
    and turns a specified attribute into a delimited string of those
    attributes.

    :param obj: String representation of object
    :param name: Name of the key in the dictionary to extract
    :param sep: What to delimit the string with
    :return: Delimited string
    """
    dicts = ast.literal_eval(obj)
    if type(dicts) is list:
        return sep.join(d[name] for d in ast.literal_eval(obj) if type(d) is dict)
    else:
        return np.NaN


def clean_meta_data(df: pd.DataFrame, in_place: bool = False) -> pd.DataFrame:
    """
    Cleans the dataframe obtained from data/movies_metadata.csv

    :param df: DataFrame object obtained from pd.read_csv()
    :param in_place: Whether or not to return the cleaned df in place or not
    :return: Cleaned pandas DataFrame object
    """
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

    # Filter problematic ids and set id to be an integer
    new_df = new_df[new_df["id"].str.isnumeric()]
    new_df["id"] = new_df["id"].astype(int)

    # Add MovieId
    new_df["MovieId"] = new_df.index + 1

    # Production Companies
    new_df["production_companies"] = new_df["production_companies"].map(lambda x: stringify(obj=x) if x is not np.NaN else np.NaN)

    # Production Countries
    new_df["production_countries"] = new_df["production_countries"].map(lambda x: stringify(obj=x) if x is not np.NaN else np.NaN)

    # Spoken Languages
    new_df["spoken_languages"] = new_df["spoken_languages"].map(lambda x: stringify(obj=x) if x is not np.NaN else np.NaN)

    return new_df


def clean_keywords(df: pd.DataFrame, in_place: bool = False) -> pd.DataFrame:
    """
    Cleans the dataframe obtained from data/keywords.csv

    :param df: DataFrame object obtained from pd.read_csv()
    :param in_place: Whether or not to return the cleaned df in place or not
    :return: Cleaned pandas DataFrame object
    """
    if not in_place:
        new_df = df.copy()
    else:
        new_df = df

    # Format keywords
    new_df["keywords"] = new_df["keywords"].map(lambda x: stringify(obj=x) if x is not np.NaN else np.NaN).astype(str)

    # Add MovieId
    new_df["MovieId"] = new_df.index + 1

    return new_df


def clean_movies(df: pd.DataFrame, in_place: bool = False) -> pd.DataFrame:
    """
    Cleans the dataframe obtained from data/movies.dat

    :param df: DataFrame object obtained from pd.read_csv()
    :param in_place: Whether or not to return the cleaned df in place or not
    :return: Cleaned pandas DataFrame object
    """
    if not in_place:
        new_df = df.copy()
    else:
        new_df = df

    # Extract the release year from the title
    new_df["Release"] = new_df["Title"].map(lambda x: re.findall(pattern=r"\d{4}", string=x)[0]).astype(int)

    # Remove the release year from the title
    new_df["Title"] = new_df["Title"].map(lambda x: re.sub(pattern=r"\s+\(\d{4}\)", string=x, repl=""))

    # Split genres into a comma delimited list
    new_df["Genres"] = new_df["Genres"].map(lambda x: ", ".join(x.split("|")))

    return new_df


def percentileMetric(df: pd.DataFrame) -> float:
    """
    This scores predictions based on the top 5% of ratings grouped by user.

    The dataframe is expected to have a "user", "predictedrating", and "actualrating" column.
    :param df: Pandas dataframe holding the scores
    :return: The mean of the actual scores
    """
    if type(df) is not pd.DataFrame:
        raise ValueError("Error: df must be a pandas dataframe")

    top5Percent = df[df.groupby("user")["predictedrating"].rank(pct=True) > 0.95]
    return top5Percent["actualrating"].values.mean()
