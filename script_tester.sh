#!/bin/bash

# Experiment settings
EXPERIMENT_NAME=Test
RESULTS_FOLDER=.
DATA_FOLDER=../data/created_data_sets
DATA_FILE=CoordinateSearchlog2.csv
CUDA_EXP=false

# Data headers
OBS_COLS="user_location_latitude user_location_longitude"
CONTEXT_COLS="hour_sin hour_cos dayofweeek_sin dayofweek_cos month_sin month_cos wind_dir_avg_sin wind_dir_avg_cos wind_speed_avg air_temperature rain_accumulation rain_duration rain_intensity"

# Regularization settings
NOISE_REG_SCHEDULER=constant
NOISE_REG_SIGMA=0.1

# Training settings
EPOCHS=10
BATCH_SIZE=50000

# Flow settings
FLOW_DEPTH=24
C_NET_DEPTH=5
C_NET_H_DIM=25
CONTEXT_N_DEPTH=5
CONTEXT_N_H_DIM=24
RICH_CONTEXT_DIM=6

python3 test2.py --experiment_name $EXPERIMENT_NAME --results_folder $RESULTS_FOLDER --data_folder $DATA_FOLDER --data_file $DATA_FILE --cuda_exp $CUDA_EXP\
                       --obs_cols $OBS_COLS --context_cols $CONTEXT_COLS \
                       --noise_reg_scheduler $NOISE_REG_SCHEDULER --noise_reg_sigma $NOISE_REG_SIGMA\
                       --epochs $EPOCHS --batch_size $BATCH_SIZE \
                       --flow_depth $FLOW_DEPTH --c_net_depth $C_NET_DEPTH --c_net_h_dim $C_NET_H_DIM --context_n_depth $CONTEXT_N_DEPTH --context_n_h_dim $CONTEXT_N_H_DIM --rich_context_dim $RICH_CONTEXT_DIM