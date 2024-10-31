from datasets import load_dataset
import pandas as pd

ds = load_dataset("BAAI/CapsFusion-120M", split="train", streaming=True)
print(next(iter(ds)))
first_1k = ds.select(range(1000))
first_1k.to_csv("first_1k.csv")
