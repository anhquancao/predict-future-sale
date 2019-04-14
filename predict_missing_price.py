import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

data = pd.read_pickle("data_test.pkl")

train = data[~data['item_price'].isna()]
test = data[data["item_price"].isna()]

columns = ["item_id", "shop_id", "date_block_num", "item_category_id"]

X_train = train[columns]
y_train = train['item_price']

X_test = test[columns]
y_test = test['item_price']

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.33, random_state=42)

d_train = lgb.Dataset(X_train, label=y_train)
d_val = lgb.Dataset(X_val, label=y_val)
watch_list = [d_train, d_val]

params = {
    'boosting_type': 'gbdt',
    'objective': 'root_mean_squared_error',
    'metric': 'l2_root',
    'num_leaf': 150,
    'learning_rate': 0.01,
    'verbose': 0,
    'early_stopping_round': 1000}
n_estimators = 5000

model = lgb.train(
    params=params,
    train_set=d_train,
    num_boost_round=n_estimators,
    valid_sets=watch_list, verbose_eval=1)
