import pandas as pd


def get_signal_baseline(df: pd.DataFrame, inversion_term: int):
    return (inversion_term - df).min().min()
