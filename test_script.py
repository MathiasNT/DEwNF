import argparse
import pickle
import os
import pandas as pd

from DEwNF.flows import ConditionalAffineCoupling2, ConditionedAffineCoupling2, ConditionalNormalizingFlowWrapper, conditional_affine_coupling2, normalizing_flow_factory, conditional_normalizing_flow_factory2
from DEwNF.utils import plot_4_contexts_cond_flow, plot_loss, sliding_plot_loss, plot_samples, plot_train_results, split_on_days
from DEwNF.samplers import RotatingTwoMoonsConditionalSampler
from DEwNF.regularizers import NoiseRegularizer, rule_of_thumb_noise_schedule, approx_rule_of_thumb_noise_schedule, square_root_noise_schedule, constant_regularization_schedule

import torch.optim as optim


def main(args):
    # cuda
    cuda_exp = args.cuda_exp == "true"

    print(cuda_exp)

    # Notebook experiment settings
    experiment_name = args.experiment_name
    experiment_results_folder = args.results_folder
    results_path = os.path.join("../", experiment_results_folder)
    data_folder = args.data_folder
    data_file = args.data_file

    # Regularization settings
    if args.noise_reg_scheduler == "constant":
        noise_reg_schedule = constant_regularization_schedule
    elif args.noise_reg_scheduler == "sqrt":
        noise_reg_schedule = square_root_noise_schedule
    elif args.noise_reg_scheduler == "rot":
        noise_reg_schedule = rule_of_thumb_noise_schedule
    else:
        noise_reg_schedule = constant_regularization_schedule

    noise_reg_sigma = args.noise_reg_sigma  # Used as sigma in rule of thumb and as noise in const

    # Data settings
    obs_cols = args.obs_cols
    context_cols = args.context_cols

    # Training settings
    epochs = args.epochs
    batch_size = args.epochs

    # Dimensions of problem
    problem_dim = len(args.obs_cols)
    context_dim = len(args.context_cols)

    # Flow settings
    flow_depth = args.flow_depth
    c_net_depth = args.c_net_depth
    c_net_h_dim = args.c_net_h_dim

    # Define context conditioner
    context_n_depth = args.context_n_depth
    context_n_h_dim = args.context_n_h_dim
    rich_context_dim = args.rich_context_dim

    settings_dict = {
        "epochs": epochs,
        "batch_size": batch_size,
        "problem_dim": problem_dim,
        "context_dim": context_dim,
        "flow_depth": flow_depth,
        "c_net_depth": c_net_depth,
        "c_net_h_dim": c_net_h_dim,
        "context_n_depth": context_n_depth,
        "context_n_h_dim": context_n_h_dim,
        "rich_context_dim": rich_context_dim,
        "obs_cols": obs_cols,
        "context_cols": context_cols
    }

    print(f"Settings:\n{settings_dict}")

    # Load data
    csv_path = os.path.join(data_folder, data_file)
    donkey_df = pd.read_csv(csv_path, parse_dates=[4, 11])

    train_dataloader, test_dataloader = split_on_days(donkey_df, obs_cols, context_cols, batch_size, cuda_exp)

    # Define stuff for reqularization
    data_size = len(train_dataloader)
    data_dim = problem_dim + context_dim

    # Define normalizing flow
    normalizing_flow = conditional_normalizing_flow_factory2(flow_depth=flow_depth,
                                                             problem_dim=problem_dim,
                                                             c_net_depth=c_net_depth,
                                                             c_net_h_dim=c_net_h_dim,
                                                             context_dim=context_dim,
                                                             context_n_h_dim=context_n_h_dim,
                                                             context_n_depth=context_n_depth,
                                                             rich_context_dim=rich_context_dim,
                                                             cuda=cuda_exp)

    # Setup Optimizer
    optimizer = optim.Adam(normalizing_flow.modules.parameters(), lr=1e-4)
    print("number of params: ", sum(p.numel() for p in normalizing_flow.modules.parameters()))

    # Setup regularization
    h = noise_reg_schedule(data_size, data_dim, noise_reg_sigma)
    noise_reg = NoiseRegularizer(discrete_dims=None, h=h, cuda=cuda_exp)
    print(f"Data size: {train_dataloader.dataset.shape}")
    print(f"Noise scale: {h}")

    # Train and test sizes
    n_train = train_dataloader.dataset.shape[0]
    n_test = test_dataloader.dataset.shape[0]
    print(f"n_train {n_train}")
    print(f"n_test {n_test}")

    # Training loop
    full_train_losses = []
    train_losses = []
    test_losses = []
    no_noise_losses = []

    for epoch in range(1, epochs + 1):

        normalizing_flow.modules.train()
        train_epoch_loss = 0
        for k, batch in enumerate(train_dataloader):
            # Add noise reg to two moons
            batch = noise_reg.add_noise(batch)
            x = batch[:, :problem_dim]
            context = batch[:, problem_dim:]

            # Condition the flow on the sampled covariate and calculate -log_prob = loss
            conditioned_flow_dist = normalizing_flow.condition(context)
            loss = -conditioned_flow_dist.log_prob(x).sum()

            # Calculate gradients and take an optimizer step
            normalizing_flow.modules.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
        full_train_losses.append(train_epoch_loss / n_train)
        print(f" Epoch {epoch} Train loss: {full_train_losses[-1]}")

        # save every 10 epoch to log and eval
        if epoch % 10 == 0 or epoch == epochs - 1:
            normalizing_flow.modules.eval()
            train_losses.append(train_epoch_loss / n_train)

            no_noise_epoch_loss = 0
            for k, batch in enumerate(train_dataloader):
                # Add noise reg to two moons
                x = batch[:, :problem_dim]
                context = batch[:, problem_dim:]

                # Condition the flow on the sampled covariate and calculate -log_prob = loss
                conditioned_flow_dist = normalizing_flow.condition(context)
                loss = -conditioned_flow_dist.log_prob(x).sum()

                no_noise_epoch_loss += loss.item()
            no_noise_losses.append(no_noise_epoch_loss / n_train)

            test_epoch_loss = 0
            for j, batch in enumerate(test_dataloader):
                # Sample covariates and use them to sample from conditioned two_moons
                x = batch[:, :problem_dim]
                context = batch[:, problem_dim:]

                # Condition the flow on the sampled covariate and calculate -log_prob = loss
                conditioned_flow_dist = normalizing_flow.condition(context)
                test_loss = -conditioned_flow_dist.log_prob(x).sum()

                test_epoch_loss += test_loss.item()
            test_losses.append(test_epoch_loss / n_test)

        # Plot Epoch resultsif epoch == epochs-1:
        if epoch == epochs - 1:
            normalizing_flow.modules.eval()
            print(f"Epoch {epoch}: train loss: {train_losses[-1]} no noise loss:{no_noise_losses[-1]} test_loss: {test_losses[-1]}")
    experiment_dict = {'train': train_losses, 'test': test_losses, 'no_noise_losses': no_noise_losses}

    results_dict = {'model': normalizing_flow, 'settings': settings_dict, 'logs': experiment_dict}

    file_name = f"{experiment_name}.pickle"
    file_path = os.path.join(results_path, file_name)

    print(f"Saving: {file_name}")
    with open(file_path, 'wb') as f:
        saved_flow = pickle.dump(results_dict, f)

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Folder args
    parser.add_argument("--experiment_name", help="Name of the experiment, used in naming outputs")
    parser.add_argument("--results_folder", help="folder for results")
    parser.add_argument("--data_folder", help="Folder with the data")
    parser.add_argument("--data_file", help="filename for the data")
    parser.add_argument("--cuda_exp", help="whether to use cuda")

    # Regularization args
    parser.add_argument("--noise_reg_scheduler", help="which noise regularization schedule should be run")
    parser.add_argument("--noise_reg_sigma", type=float, help="Sigma value for the noise reg schedule")

    # Data args
    parser.add_argument("--obs_cols", nargs="+", help="The column names for the observation data")
    parser.add_argument("--context_cols", nargs="+", help="The headers for the context data")

    # Training args
    parser.add_argument("--epochs", type=int, help="number of epochs")
    parser.add_argument("--batch_size", type=int, help="batch size for training")

    # flow args
    parser.add_argument("--flow_depth", type=int, help="number of layers in flow")
    parser.add_argument("--c_net_depth", type=int, help="depth of the conditioner")
    parser.add_argument("--c_net_h_dim", type=int, help="hidden dimension of the conditioner")
    parser.add_argument("--context_n_depth", type=int, help="depth of the conditioning network")
    parser.add_argument("--context_n_h_dim", type=int, help="hidden dimension of the context network")
    parser.add_argument("--rich_context_dim", type=int, help="dimension of the generated rich context")

    args = parser.parse_args()

    main(args)
