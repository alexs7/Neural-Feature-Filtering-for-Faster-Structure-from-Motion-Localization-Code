# Neural Feature Filtering for Faster Structure from Motion Localization

### The code for Neural Feature Filtering for Faster Structure from Motion Localization

The files themselves will contain comments on how to use them.

#### To get data for Match and No Match, run the following commands:

- create_universal_models.py
- get_points_3D_mean_desc_ml_mnm.py
- create_training_data_and_train_for_match_no_match.py

#### To get data for Predicticting Matchability, run the following commands:

- create_training_data_predicting_matchability.py
- train_for_predicting_matchability.py

#### To get data for Neural Filtering, run the following commands:

- create_nf_training_data.py
- train_for_nf.py

#### Benchmarking the models:

- learned_models_benchmarks.py #will generate model statistics
- learned_models_pose_data.py #will generate pose data statistics

#### Notes:

In most scripts you have to pass, 'HGE' or 'CMU' or 'RetailShop' to specify the dataset