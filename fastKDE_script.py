import argparse
import pickle
import os
import pandas as pd

from DEwNF.utils import get_split_idx_on_day
from time import time

import numpy as np
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional
from sklearn.preprocessing import StandardScaler
from fastkde.fastKDE import fastKDE

def main(args):
    # Notebook experiment settings
    experiment_name = args.experiment_name
    experiment_results_folder = args.results_folder
    results_path = os.path.join("../", experiment_results_folder)
    data_folder = args.data_folder
    data_file = args.data_file
    extra_data_file = args.extra_data_file

    file_name_test = f"{experiment_name}_test.pickle"
    file_path_test = os.path.join(results_path, file_name_test)

    print(f"Saving: {file_name_test}")
    with open(file_path_test, 'wb') as f:
        pickle.dump("test", f)

    # Data settings
    obs_cols = ['user_location_latitude', 'user_location_longitude']
    semisup_context_cols = ['hour_sin', 'hour_cos']
    joint_cols = obs_cols + semisup_context_cols


    # Load data
    csv_path = os.path.join(data_folder, data_file)
    donkey_df = pd.read_csv(csv_path, parse_dates=[4, 11])

    csv_path = os.path.join(data_folder, extra_data_file)
    extra_df = pd.read_csv(csv_path, parse_dates=[4, 12])

    # Data prep
    train_idx, test_idx = get_split_idx_on_day(donkey_df)

    # Create full data
    test_data = donkey_df.loc[test_idx, joint_cols]
    train_data = pd.concat((donkey_df.loc[train_idx, joint_cols], extra_df.loc[:, joint_cols]))

    # Normalize data
    obs_scaler = StandardScaler().fit(train_data)
    scaled_train_data = obs_scaler.transform(train_data)

    scaled_test_data = obs_scaler.transform(test_data)

    fastKDE = fastKDE(scaled_train_data.T)
    print(fastKDE)
    results_dict = {'model': fastKDE }

    file_name = f"{experiment_name}.pickle"
    file_path = os.path.join(results_path, file_name)

    print(f"Saving: {file_name}")
    with open(file_path, 'wb') as f:
        pickle.dump(results_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Folder args
    parser.add_argument("--experiment_name", help="Name of the experiment, used in naming outputs")
    parser.add_argument("--results_folder", help="folder for results")
    parser.add_argument("--data_folder", help="Folder with the data")
    parser.add_argument("--data_file", help="filename for the data")
    parser.add_argument("--extra_data_file", help="file name for the extra data for semisupervised learning")

    args = parser.parse_args()

    start = time()
    main(args)
    end = time()
    print(f"finished elapsed time: {end-start}")

