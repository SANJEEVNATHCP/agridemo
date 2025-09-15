import pandas as pd
import numpy as np

def create_crop_dataset():
    df = pd.DataFrame({
        "crop":["wheat","rice","maize","cotton"]*10,
        "temp_max":np.random.randint(25,40,40),
        "temp_min":np.random.randint(15,25,40),
        "rainfall":np.random.randint(50,200,40),
        "month":np.random.randint(1,12,40),
        "demand":np.random.randint(1,10,40),
        "price":np.random.randint(1000,3000,40)
    })
    df.to_csv("crop_prices.csv",index=False)
    print("✅ Dummy dataset saved as crop_prices.csv")

if __name__=="main_":
    create_crop_dataset()