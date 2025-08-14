import pandas as pd
from src.data.data_preprocessor import build_preprocessor

def test_build_preprocessor():
    df = pd.DataFrame({
        "LotArea": [8450, 9600, None],
        "Street": ["Pave", "Pave", "Grvl"],
        "SalePrice": [208500, 181500, 223500]
    })
    pre, nums, cats = build_preprocessor(df, "SalePrice")
    assert "LotArea" in nums and "Street" in cats
