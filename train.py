import pandas as pd

language = "en"
data_train = pd.concat(
    [
        pd.read_excel(f"/data/readme/dataset/{language}/readme_{language}_train.xlsx"),
        pd.read_excel(f"/data/readme/dataset/{language}/readme_{language}_val.xlsx"),
    ],
    axis=0,
)
data_val = pd.read_excel(f"/data/readme/dataset/{language}/readme_{language}_test.xlsx")
