from implicit.nearest_neighbours import CosineRecommender, TFIDFRecommender


class UserKnn_model_conf():
    # models: CosineRecommender or TFIDFRecommender
    model = TFIDFRecommender()
    weight_path = 'data/weights/userknn_TFIDF.dill'
    save_reco_df_path = 'data/offline_reco_df/userknn_TFIDF'
    N_recs = 10
    online = False
    blend_threshold = 3
    # train test split
    # test = 4 days (startdate_test = lastdate - 12 days)
    n_folds = 3
    unit = "D"
    n_units = 4
    periods = 2
    # data
    dataset_path = 'data/kion_train'


class Popular_model_conf:
    save_reco_df_path = 'data/offline_reco_df/popular'
    dataset_path = 'data/kion_train'
    N_recs = 10
