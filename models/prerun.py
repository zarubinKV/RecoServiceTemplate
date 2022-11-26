import pandas as pd
from implicit.nearest_neighbours import CosineRecommender
import warnings
warnings.filterwarnings("ignore")
from rectools.dataset import Dataset
from rectools import Columns
from rectools.models.popular import PopularModel
from models.userknn.userknn import UserKnn
from rectools.model_selection import TimeRangeSplit

interactions = pd.read_csv('data/kion_train/interactions.csv')
users = pd.read_csv('data/kion_train/users.csv')
items = pd.read_csv('data/kion_train/items.csv')

# rename columns, convert timestamp
interactions.rename(columns={'last_watch_dt': Columns.Datetime,
                            'total_dur': Columns.Weight},
                    inplace=True)

interactions['datetime'] = pd.to_datetime(interactions['datetime'])

max_date = interactions['datetime'].max()
min_date = interactions['datetime'].min()

# setting for cv
n_folds = 7
unit = "W"
n_units = 1

last_date = interactions[Columns.Datetime].max().normalize()
start_date = last_date - pd.Timedelta(n_folds * n_units + 1, unit=unit)

# train test split
# test = last 1 week
n_folds = 1
unit = "W"
n_units = 1
periods = n_folds + 1
freq = f"{n_units}{unit}"
last_date = interactions[Columns.Datetime].max().normalize()
start_date = last_date - pd.Timedelta(n_folds * n_units + 1, unit=unit)

date_range = pd.date_range(start=start_date, periods=periods, freq=freq,
                           tz=last_date.tz)
# generator of folds
cv = TimeRangeSplit(
    date_range=date_range,
    filter_already_seen=True,
    filter_cold_items=True,
    filter_cold_users=True,
)
(train_ids, test_ids, fold_info) = cv.split(interactions, collect_fold_stats=True).__next__()

train = interactions.loc[train_ids]
test = interactions.loc[test_ids]
userknn_model = UserKnn(model=CosineRecommender(), N_users=50)
userknn_model.load_weight(train, 'data/weights/userknn.dill')

userknn_reco_df = userknn_model.predict(test)

dataset = Dataset.construct(
    interactions_df=interactions,
    user_features_df=None,
    item_features_df=None,
)

pop = PopularModel()
pop.fit(dataset)
popular_reco_df = pop.recommend(
    dataset.user_id_map.external_ids[:1],
    dataset=dataset,
    k=10,
    filter_viewed=False  # True - throw away some items for each user
)
