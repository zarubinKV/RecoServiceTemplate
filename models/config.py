
class UserKnn_model:
    weight_path = 'data/weights/userknn.dill'
    save_reco_df_path = 'data/offline_reco_df/userknn'
    dataset_path = 'data/kion_train'
    N_recs = 10
    online = False

class Popular_model:
    save_reco_df_path = 'data/offline_reco_df/popular'
    dataset_path = 'data/kion_train'
    N_recs = 10
