import pandas as pd
from implicit.nearest_neighbours import CosineRecommender
from rectools.dataset import Dataset
from rectools import Columns
from rectools.models.popular import PopularModel
from models.userknn.userknn import UserKnn
from rectools.model_selection import TimeRangeSplit
from models.config import UserKnn_model, Popular_model

USERKNN = {
    'model_loaded': False,
    'model': None,
    'reco_df': None,
}

POPULAR = {
    'model_loaded': False,
    'reco_df': None,
}


def load_data():
    global train
    global interactions
    interactions = pd.read_csv(
        '{0}/interactions.csv'.format(UserKnn_model.dataset_path))
    users = pd.read_csv('{0}/users.csv'.format(UserKnn_model.dataset_path))
    items = pd.read_csv('{0}/items.csv'.format(UserKnn_model.dataset_path))
    # rename columns, convert timestamp
    interactions.rename(columns={'last_watch_dt': Columns.Datetime,
                                 'total_dur': Columns.Weight},
                        inplace=True)
    interactions['datetime'] = pd.to_datetime(interactions['datetime'])
    # train test split
    # test = 4 days (startdate_test = lastdate - 12 days)
    n_folds = 3
    unit = "D"
    n_units = 4
    periods = 2
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
    (train_ids, test_ids, fold_info) = cv.split(
        interactions,
        collect_fold_stats=True,
    ).__next__()
    train = interactions.loc[train_ids]


def load_userknn():
    global USERKNN
    if not USERKNN['model_loaded']:
        USERKNN['model_loaded'] = True
        if UserKnn_model.online:
            USERKNN['model'] = UserKnn(model=CosineRecommender(), N_users=50)
            USERKNN['model'].load_weight(
                train,
                UserKnn_model.weight_path,
            )
        else:
            USERKNN['reco_df'] = pd.read_csv(
                UserKnn_model.save_reco_df_path,
                encoding='utf-8',
            )


def load_popular():
    global POPULAR
    if not POPULAR['model_loaded']:
        POPULAR['reco_df'] = pd.read_csv(
            Popular_model.save_reco_df_path,
            encoding='utf-8',
        )


interactions = None
train = None
load_data()
load_userknn()
load_popular()
interactions = train
