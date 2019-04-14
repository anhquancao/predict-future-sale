import pandas as pd

item_categories = pd.read_csv("data/item_categories_en.csv", header=0)

cats = item_categories['item_category_name'].str.split(
    "-", n=1, expand=True).iloc[:, 0].str.strip()

item_categories['cat'] = cats

item_categories.to_csv("data/item_categories_en.csv")
