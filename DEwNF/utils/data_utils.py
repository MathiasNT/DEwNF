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
    scaled_train_data = torch.cat((torch.tensor(scaled_train_obs), torch.tensor(scaled_train_context)), dim=1).type(torch.FloatTensor)

    # Transform test data
    scaled_test_obs = obs_scaler.transform(df.loc[test_idx, obs_cols])
    scaled_test_context = context_scaler.transform(df.loc[test_idx, context_cols])
    scaled_test_data = torch.cat((torch.tensor(scaled_test_obs), torch.tensor(scaled_test_context)), dim=1).type(torch.FloatTensor)

    # Send to cuda if necessary
    if cuda_exp:
        scaled_train_data = scaled_train_data.cuda()
        scaled_test_data = scaled_test_data.cuda()

    # Wrap in dataloaders to take care of batching
    train_dataloader = DataLoader(scaled_train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(scaled_test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader, obs_scaler, context_scaler


def split_synthetic(df, batch_size, data_size, cuda_exp):
    # Calculate split id
    split_id = int(data_size * 0.8)

    # Take subset of data and split
    df = df[:data_size]
    train_df = df[:split_id]
    test_df = df[split_id:]

    # Fit transform
    obs_scaler = StandardScaler().fit(train_df.iloc[:, 0:2])
    context_scaler = StandardScaler().fit(train_df.iloc[:, 2:])

    # Transform training data
    scaled_train_obs = obs_scaler.transform(train_df.iloc[:, 0:2])
    scaled_train_context = context_scaler.transform(train_df.iloc[:, 2:])
    scaled_train_data = torch.cat((torch.tensor(scaled_train_obs), torch.tensor(scaled_train_context)), dim=1).type(torch.FloatTensor)

    # Transform test data
    scaled_test_obs = obs_scaler.transform(test_df.iloc[:, 0:2])
    scaled_test_context = context_scaler.transform(test_df.iloc[:, 2:])
    scaled_test_data = torch.cat((torch.tensor(scaled_test_obs), torch.tensor(scaled_test_context)), dim=1).type(torch.FloatTensor)

    # Send to cuda if necessary
    if cuda_exp:
        scaled_train_data = scaled_train_data.cuda()
        scaled_test_data = scaled_test_data.cuda()

    # Wrap in dataloaders to take care of batching
    train_dataloader = DataLoader(scaled_train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(scaled_test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader
