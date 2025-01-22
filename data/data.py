# this downloads the datasets into your computer

"""import kagglehub

# Download latest version
path = kagglehub.dataset_download("hassan06/nslkdd")

print("Path to dataset files:", path)
"""
import os
from dotenv import load_dotenv
import pandas as pd 

load_dotenv()

KDD_TRAIN = os.getenv("KDD_TRAIN")

df = pd.read_csv(KDD_TRAIN, sep=',')

print(df)



