import pandas as pd
from sklearn.datasets import load_boston

def get_boston_df():
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=(c.lower() for c in boston.feature_names))
    df['price'] = boston.target
    return df