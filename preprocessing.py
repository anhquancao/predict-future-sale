import pandas as pd
import numpy as np


def create_monthly_test_data(data_train):
    # Read in test data
    test = pd.read_csv("data/test.csv")
    data_test = test.drop(['ID'], axis=1)
    data_test = data_test[["item_id", 'shop_id']]
    data_test['date_block_num'] = 34
    data_test['month'] = 11
    data_test['quarter'] = 4

    # construct the test price using last (shop_id, item_id) price
    shop_item_prices = data_train.groupby(['item_id', 'shop_id'])['item_price']\
        .last()\
        .reset_index()

    data_test = pd.merge(
        data_test,
        shop_item_prices,
        left_on=['shop_id', 'item_id'],
        right_on=['shop_id', 'item_id'],
        how='left')

    # construct the test price using item_id average price
    item_prices = data_train.groupby(
        "item_id")['item_price'].mean().reset_index()
    t = pd.merge(data_test, item_prices, left_on=[
                 'item_id'], right_on=['item_id'], how="left")
    prices = []
    for index, row in t.iterrows():
        if np.isnan(row['item_price_x']):
            prices.append(row['item_price_y'])
        else:
            prices.append(row['item_price_x'])
    data_test['item_price'] = prices
    return data_test


def create_monthly_train_data():
    sale_train = pd.read_csv("data/sales_train_v2.csv")
    sale_train['date'] = pd.to_datetime(sale_train['date'], format='%d.%m.%Y')
    sale_train['month'] = sale_train['date'].dt.month
    sale_train['quarter'] = np.ceil(sale_train['month'] / 3)

    data = sale_train.groupby(
        ['item_id', 'shop_id', 'date_block_num', "month", "quarter"]).agg({
            "item_price": "mean",
            "item_cnt_day": "sum"
        }).reset_index()

    return data


def add_item_category(data):
    items = pd.read_csv("data/items.csv")
    items.drop(["item_name"], axis=1, inplace=True)

    item_categories = pd.read_csv("data/item_categories_en.csv")
    item_categories.drop(['item_category_name'], axis=1, inplace=True)

    data = pd.merge(
        data,
        items,
        left_on=['item_id'],
        right_on=['item_id'])

    data = pd.merge(
        data,
        item_categories,
        left_on=['item_category_id'],
        right_on=['item_category_id'])

    return data


def add_shop_city(data):
    shops = pd.read_csv("data/shops_en.csv")
    shops.drop(['shop_name'], axis=1, inplace=True)
    return pd.merge(
        data,
        shops,
        left_on=['shop_id'],
        right_on=['shop_id']
    )


def combine_data():
    data_train = create_monthly_train_data()
    data_test = create_monthly_test_data(data_train)
    data_test['item_cnt_day'] = 0
    data_train = data_train[data_test.columns]

    data = pd.concat([data_train, data_test])
    data = add_item_category(data)
    data = add_shop_city(data)

    data['item_cnt_month'] = data['item_cnt_day']
    data.drop(['item_cnt_day'], axis=1, inplace=True)

    data.to_pickle("data/data.pkl")

    return data


data = combine_data()
