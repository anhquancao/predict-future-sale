import pandas as pd


data = pd.read_pickle("data/data.pkl")


def rank_encode(data, field):
    map_code = {}
    cats = data.groupby(field).size().sort_values().reset_index()[field]
    for i in range(len(cats)):
        map_code[cats[i]] = i
    return map_code


map_category = rank_encode(data, 'cat')
data["cat"] = data['cat'].apply(lambda x: map_category[x])

map_city = rank_encode(data, 'city')
data["city"] = data['city'].apply(lambda x: map_city[x])

data.to_pickle("data/features.pkl")
