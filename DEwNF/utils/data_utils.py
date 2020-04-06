from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader


def split_on_days(df, obs_cols, context_cols, batch_size, cuda_exp):
    # Split data into test and train sets based on the days in the data
    unique_days = df.obs_day.unique()
    train_days, test_days = train_test_split(unique_days, test_size=0.2, random_state=42)
    train_idx = df.index[df.obs_day.isin(train_days)]
    test_idx = df.index[df.obs_day.isin(test_days)]
    df = df.drop('obs_day', axis=1)

    # Normalize data and send to cuda
    # Fit transformation only on the train data
    obs_scaler = StandardScaler().fit(df.loc[train_idx, obs_cols])
    context_scaler = StandardScaler().fit(df.loc[train_idx, context_cols])

    # Transform training data
    scaled_train_obs = obs_scaler.transform(df.loc[train_idx, obs_cols])
    scaled_train_context = context_scaler.transform(df.loc[train_idx, context_cols])
    scaled_trained_data = torch.cat((torch.tensor(scaled_train_obs), torch.tensor(scaled_train_context)), dim=1).type(torch.FloatTensor)

    # Transform test data
    scaled_test_obs = obs_scaler.transform(df.loc[test_idx, obs_cols])
    scaled_test_context = context_scaler.transform(df.loc[test_idx, context_cols])
    scaled_test_data = torch.cat((torch.tensor(scaled_test_obs), torch.tensor(scaled_test_context)), dim=1).type(torch.FloatTensor)

    # Send to cuda if necessary
    if cuda_exp:
        scaled_trained_data = scaled_trained_data.cuda()
        scaled_test_data = scaled_test_data.cuda()

    # Wrap in dataloaders to take care of batching
    train_dataloader = DataLoader(scaled_trained_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(scaled_test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


def split_synthetic(df, batch_size, data_size, cuda_exp):
    # Take subset of data
    df = df[:data_size]

    # Normalize data and send to cuda
    obs_scaler = StandardScaler().fit(df.iloc[:, 0:2])
    context_scaler = StandardScaler().fit(df.iloc[:, 2:])
    scaled_obs = obs_scaler.transform(df.iloc[:, 0:2])
    scaled_context = context_scaler.transform(df.iloc[:, 2:])
    scaled_data = torch.cat((torch.tensor(scaled_obs), torch.tensor(scaled_context)), dim=1).type(torch.FloatTensor)
    data_tensors = scaled_data

    # Split into test and train and put in DataLoader
    data_len = data_tensors.shape[0]
    data_split_id = int(data_len * 0.8)
    train_dataloader = DataLoader(data_tensors[:data_split_id], batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(data_tensors[data_split_id:], batch_size=batch_size)
    return train_dataloader, test_dataloader
