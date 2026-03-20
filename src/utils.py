import pandas as pd

def load_csv(path, compression=None):
    """
    Load processed CSV files used in the EV project
    """
    return pd.read_csv(path, compression=compression)

