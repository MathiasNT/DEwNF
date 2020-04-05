import argparse
import pickle
import os
import pandas as pd
from tqdm import tqdm

from DEwNF.flows import ConditionalAffineCoupling2, ConditionedAffineCoupling2, ConditionalNormalizingFlowWrapper, conditional_affine_coupling2, normalizing_flow_factory, conditional_normalizing_flow_factory2
from DEwNF.utils import plot_4_contexts_cond_flow, plot_loss, sliding_plot_loss, plot_samples, plot_train_results, split_on_days
from DEwNF.samplers import RotatingTwoMoonsConditionalSampler
from DEwNF.regularizers import NoiseRegularizer, rule_of_thumb_noise_schedule, approx_rule_of_thumb_noise_schedule, square_root_noise_schedule, constant_regularization_schedule

import torch.optim as optim
import torch


def main(args):
    # cuda
    cuda_exp = args.cuda_exp == "true"

    print(cuda_exp)

    tensortest = torch.tensor([1, 1]).cuda()
    print(tensortest)
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
