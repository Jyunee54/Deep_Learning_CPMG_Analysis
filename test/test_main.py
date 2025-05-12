import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import glob
import sys
from imports.utils import *
from imports.models import *
from HPC import *
from imports.adabound import AdaBound
import time
import argparse
import pickle


# ----------------------------- #
# 실행 설정 (기존 코드 그대로 유지)
# ----------------------------- #
CUDA_DEVICE = 0
N_PULSE = 32
IMAGE_WIDTH = 10
TIME_RANGE_32  = 7000
TIME_RANGE_256  = 0
EXISTING_SPINS = 0
EVALUATION_ALL = 0

target_side_distance = 3000
A_init  = 10000
A_final = 20000
A_step  = 200
A_range = 200
B_init  = 20000
B_final = 80000
noise_scale = 0.5
zero_scale = 0.05
SAVE_DIR_NAME = "./data/results/"
is_CNN = 0
is_remove_model_index = 0

model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)

# ✅ 핵심 실행부를 __main__ 블록으로 감쌈
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    tic = time.time()

    args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final,
            A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists,
            target_side_distance, is_CNN, is_remove_model_index)

    hpc_model = HPC_Model(*args)
    total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
    hpc_model.close_pool()

    predicted_periods = return_filtered_A_lists_wrt_pred(np.array(total_deno_pred_list[1,:]), np.array(total_A_lists), 0.8)
    np.save('./data/results/predicted_periods.npy', predicted_periods)


    print("predicted periods:", predicted_periods)


    regression_args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final,
                       A_step, A_range, B_init, B_final, zero_scale, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN)

    regression_model = Regression_Model(*regression_args)

    regression_results_all = []


    def split_grouped_periods(periods, group_size=5):
        result = []
        for group in periods:
            for i in range(0, len(group), group_size):
                result.append(group[i:i + group_size])
        return result


    split_predicted_periods = split_grouped_periods(predicted_periods, group_size=5)
    print("split_predicted_periods:", split_predicted_periods)

    for i, subgroup in enumerate(split_predicted_periods):
        print(f"[INFO] Running regression on subgroup {i + 1}: A {subgroup[0]} to {subgroup[-1]}")

        try:
            regression_results = regression_model.estimate_specific_AB_values([subgroup])
            if regression_results:
                regression_results_all.extend(regression_results)
            else:
                print(f"[WARNING] Subgroup {i + 1} returned no results.")
        except Exception as e:
            print(f"[ERROR] Subgroup {i + 1} failed with error: {e}")

    # 결과 저장
    with open('./data/results/regression_results.pkl', 'wb') as f:
        pickle.dump(regression_results_all, f)
    print('✅ 회귀 완료. 전체 결과 저장됨.')
    print('총 실행 시간:', time.time() - tic)
    print(regression_results_all)
