import pickle
from pathlib import Path
import pandas as pd

def concat_l(l):
    out = pd.concat(
        [df for df in l if not df.empty],
        ignore_index=True
    ) if any(not df.empty for df in l) else pd.DataFrame()
    return out


def load_pickle(pickle_path: Path):

    pickle_path = Path(pickle_path)

    with open(pickle_path, "rb") as f:
        out = pickle.load(f)

    return out

def save_pickle(obj, pickle_path: Path):

    pickle_path = Path(pickle_path)
    pickle_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pickle_path, "wb") as f:
        pickle.dump(obj, f)
    
    return None