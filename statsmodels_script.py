import argparse
import pickle
import os
import pandas as pd

from DEwNF.utils import get_split_idx_on_day
from time import time

import numpy as np
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional
from sklearn.preprocessing import StandardScaler

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
    obs_cols = args.obs_cols

    # Load data
    csv_path = os.path.join(data_folder, data_file)
    donkey_df = pd.read_csv(csv_path, parse_dates=[4, 11])

    csv_path = os.path.join(data_folder, extra_data_file)
    extra_df = pd.read_csv(csv_path, parse_dates=[4, 12])

    # Data prep
    train_idx, test_idx = get_split_idx_on_day(donkey_df)

    # Create full data
    test_data = donkey_df.loc[test_idx, obs_cols]
    train_data = pd.concat((donkey_df.loc[train_idx, obs_cols], extra_df.loc[:, obs_cols]))

    # Normalize data
    obs_scaler = StandardScaler().fit(train_data)
    scaled_train_data = obs_scaler.transform(train_data)

    # Create conditional variable
    hours = pd.concat((donkey_df.loc[train_idx, :], extra_df.loc[:, :])).merge_date.dt.hour.values
    hours = np.expand_dims(hours, 1)

    scaled_test_data = obs_scaler.transform(test_data)

    statsmods = KDEMultivariateConditional(endog=scaled_train_data, exog=hours, indep_type='o', dep_type='cc',
                                           bw='cv_ml')

    results_dict = {'model': statsmods, }

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


    # Data args
    parser.add_argument("--obs_cols", nargs="+", help="The column names for the observation data")

    args = parser.parse_args()

    start = time()
    main(args)
    end = time()
    print(f"finished elapsed time: {end-start}")

