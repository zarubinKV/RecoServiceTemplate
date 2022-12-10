import dill
import pandas as pd

from rectools import Columns
from rectools.model_selection import TimeRangeSplit

from config.config_models import UserKnn_model_conf, Popular_model_conf

USERKNN = {
    'model_loaded': False,
    'model': None,
    'reco_df': None,
}

POPULAR = {
    'model_loaded': False,
    'reco_df': None,
}

interactions = None

train = None


def cv_generate():
    global interactions
    # train test split
    n_folds = UserKnn_model_conf.n_folds
    unit = UserKnn_model_conf.unit
    n_units = UserKnn_model_conf.n_units
    periods = UserKnn_model_conf.periods
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
    return cv.split(interactions, collect_fold_stats=True).__next__()


def load_data():
    global train
    global interactions
    interactions = pd.read_csv(
        '{0}/interactions.csv'.format(UserKnn_model_conf.dataset_path))
    users = pd.read_csv('{0}/users.csv'.format(UserKnn_model_conf.dataset_path))
    items = pd.read_csv('{0}/items.csv'.format(UserKnn_model_conf.dataset_path))
    # rename columns, convert timestamp
    interactions.rename(columns={'last_watch_dt': Columns.Datetime,
                                 'total_dur': Columns.Weight},
                        inplace=True)
    interactions['datetime'] = pd.to_datetime(interactions['datetime'])
    (train_ids, test_ids, fold_info) = cv_generate()
    train = interactions.loc[train_ids]


def load_userknn():
    global USERKNN
    if not USERKNN['model_loaded']:
        USERKNN['model_loaded'] = True
        if UserKnn_model_conf.online:
            with open(UserKnn_model_conf.weight_path, 'rb') as f:
                USERKNN['model'] = dill.load(f)
        else:
            USERKNN['reco_df'] = pd.read_csv(
                UserKnn_model_conf.save_reco_df_path,
                encoding='utf-8',
            )


def load_popular():
    global POPULAR
    if not POPULAR['model_loaded']:
        POPULAR['reco_df'] = pd.read_csv(
            Popular_model_conf.save_reco_df_path,
            encoding='utf-8',
        )

def main():
    global interactions
    global train
    load_data()
    load_userknn()
    load_popular()
    interactions = train


