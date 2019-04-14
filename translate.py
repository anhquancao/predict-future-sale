from googletrans import Translator
import pandas as pd


def convert(str):
    translator = Translator()
    return translator.translate(str, dest='en', src='ru').text


def translate(data, column):
    data[column] = data[column].apply(convert)
    return data


# item_categories = pd.read_csv("item_categories.csv")
# item_categories = translate(item_categories, "item_category_name")
# item_categories.to_csv("item_categories_en.csv", index=False)

shops = pd.read_csv("shops.csv")
shops = translate(shops, "shop_name")
shops.to_csv("shops_en.csv", index=False)
