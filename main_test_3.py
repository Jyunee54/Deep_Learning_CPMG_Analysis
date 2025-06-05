import matplotlib.pyplot as plt
import numpy as np
from nbclient.client import timestamp
from torch import dtype

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
import multiprocessing
from datetime import datetime


# ----------------------------- #
# 실행 설정 (기존 코드 그대로 유지)
# ----------------------------- #
CUDA_DEVICE = 0
N_PULSE = 32
IMAGE_WIDTH = 10
TIME_RANGE_32  = 1000
TIME_RANGE_256  = 0
EXISTING_SPINS = 0
EVALUATION_ALL = 0

target_side_distance = 1000
A_init  = -50000
A_final = 49800
A_step  = 200
A_range = 200
B_init  = 1500
B_final = 50000
noise_scale = 0.5
zero_scale = 0.5
SAVE_DIR_NAME = "./data/results/"
is_CNN = 0
is_remove_model_index = 0

# ✅ 핵심 실행부를 __main__ 블록으로 감쌈
if __name__ == '__main__':
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')
    model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)

    # HPC 모델
    # tic = time.time()
    # args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final, A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
    #
    # hpc_model = HPC_Model(*args)
    # total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
    # print('Total computational time:', time.time() - tic)
    # hpc_model.close_pool()
    #
    # tic = time.time()
    # predicted_periods = return_filtered_A_lists_wrt_pred(np.array(total_deno_pred_list[1,:]), np.array(total_A_lists), 0.8)
    #
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # filename = f"hpc/predicted_periods_{timestamp}.npy"
    # np.save(SAVE_DIR_NAME + filename, np.array(predicted_periods, dtype=object))
    # print("✅ HPC 결과를 저장했습니다.")

    # HPC 예측
    # tic = time.time()
    # args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final, A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
    #
    # hpc_model = HPC_Model(*args)
    # total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_predict()
    # print('Total computational time:', time.time() - tic)
    # hpc_model.close_pool()
    #
    # tic = time.time()
    # predicted_periods = return_filtered_A_lists_wrt_pred(np.array(total_deno_pred_list[1,:]), np.array(total_A_lists), 0.8)
    #
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # filename = f"hpc/predicted_periods_{timestamp}.npy"
    # np.save(SAVE_DIR_NAME + filename, np.array(predicted_periods, dtype=object))
    # print("✅ HPC 결과를 저장했습니다.")
    # print("Total computational time:", time.time() - tic)
    # print(predicted_periods)


    # Regression 모델
    # periods에 대해 핵스핀 갯 탐지
    # tic = time.time()
    # predicted_periods = np.load(SAVE_DIR_NAME + 'hpc/predicted_periods_20250605-153942.npy', allow_pickle=True)
    # zero_scale = 0.5
    # regression_args = (
    # CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final,
    # A_step, A_range, B_init, B_final, zero_scale, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN)
    #
    # regression_model = Regression_Model(*regression_args)
    # regression_results = regression_model.estimate_specific_AB_values(predicted_periods)
    # regression_model.close_pool()
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # filename = f"/spin_num/regression_results_with_spin_nums_{timestamp}.npy"
    # np.save(SAVE_DIR_NAME + filename, np.array(regression_results, dtype=object))
    # print("✅ Regression 결과를 저장했습니다.")
    # print('Total computational time:', time.time() - tic)

    # # predicted_periods와 핵스핀 개수로 A, B 파라미터 예측 필요
    zero_scale = 0.5
    regression_args = (
        CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final,
        A_step, A_range, B_init, B_final, zero_scale, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance,
        is_CNN)
    regression_model = Regression_Model(*regression_args)

    regression_results = np.load(SAVE_DIR_NAME + 'spin_num/regression_results_with_spin_nums_20250605-171329.npy', allow_pickle=True)
    predicted_periods = np.load(SAVE_DIR_NAME + 'hpc/predicted_periods_20250605-154909.npy', allow_pickle=True)

    A_lists = return_the_number_of_spins(predicted_periods, regression_results)
    regression_AB_results = regression_model.estimate_specific_AB_values_with_the_number_of_spins(A_lists)
    regression_model.close_pool()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"/regression/regression_AB_results_{timestamp}.npy"
    regression_AB_results = np.array(regression_AB_results, dtype=object)
    print("✅ Regression 결과를 저장했습니다.")
    np.save(SAVE_DIR_NAME + filename, regression_AB_results)

    # B 값이 15000 이상인 하이퍼핀 파라미터만 필터링해서 저장
    # filtered_results = [res for res in regression_results if res[1] > 15000]  # res = [A, B] 형태라고 가정
    # filtered_results = np.array(filtered_results)
    # save_path = SAVE_DIR_NAME + 'predicted_results_N32_B15000above.npy'
    # np.save(save_path, filtered_results)
    # print(f"✅ B > 15000인 결과를 {save_path} 에 저장했습니다.")

    # 여기서 얻은 것 중에, B=15000보다 큰 리스트를 저장해놓음. --> 아래 경로로.
    # np.load(SAVE_DIR_NAME + 'predicted_results_N32_B15000above.npy')

    # EXISTING_SPINS = 1
    # # B_init, B_final = 6000, 12000
    # # model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)
    # args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final,
    #         A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
    # total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
    #
    # # total_raw_pred_list, total_deno_pred_list 이결과를 가지고 개수를 파악
    # regression_model = Regression_Model(*args)
    # # regression_model.estimate_the_number_of_spins(A_lists)
    # # regression_model.estimate_specific_AB_values(A_lists_with_the_number_of_spins)
    # A_lists = return_the_number_of_spins(predicted_periods, regression_results)
    # regression_AB_results = regression_model.estimate_specific_AB_values(A_lists)
    # regression_model.close_pool()
    #
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # filename = f"/regression/regression_AB_results_{timestamp}.npy"
    # regression_AB_results = np.array(regression_AB_results)
    # print("✅ Regression 결과를 저장했습니다.")
    # np.save(SAVE_DIR_NAME + filename, regression_AB_results)




    # if EVALUATION_ALL:
    #     EXISTING_SPINS = 0
    #     A_init, A_final, B_init, B_final, target_side_distance = -50000, 50000, 12000, 80000, 50
    #     model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)
    #     args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final,
    #             A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
    #
    #     hpc_model = HPC_Model(*args)
    #     total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
    #     predicted_periods = return_filtered_A_lists_wrt_pred(total_deno_pred_list[1,:], np.array(total_A_lists))
    #     np.save(SAVE_DIR_NAME + "predicted_periods.npy", np.array(predicted_periods, dtype=object))
    #
    #     regression_args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final,
    #                         A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN)
    #     regression_model = Regression_Model(*regression_args)
    #     regression_results = regression_model.estimate_specific_AB_values(predicted_periods)
    #
    #     # B 값이 15000 이상인 하이퍼핀 파라미터만 필터링해서 저장
    #     filtered_results = [res for res in regression_results if res[1] > 15000]  # res = [A, B] 형태라고 가정
    #     filtered_results = np.array(filtered_results)
    #
    #     save_path = SAVE_DIR_NAME + 'predicted_results_N32_B15000above.npy'
    #     np.save(save_path, filtered_results)
    #     print(f"✅ B > 15000인 결과를 {save_path} 에 저장했습니다.")
    #
    #     # total_raw_pred_list, total_deno_pred_list 이 결과를 가지고 개수를 파악.
    #     # regression_model = Regression_Model(*args)
    #     # regression_model.estimate_the_number_of_spins(A_lists)
    #     # regression_model.estimate_specific_AB_values(A_lists_with_the_number_of_spins)
    #     # 여기서 얻은 것 중에, B=15000보다 큰 리스트를 저장해놓음. --> 아래 경로로.
    #     # np.load(SAVE_DIR_NAME + 'predicted_results_N32_B15000above.npy')
    #
    #     EXISTING_SPINS = 1
    #     B_init, B_final = 6000, 12000
    #     model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)
    #     args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final,
    #             A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
    #     total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
    #     # total_raw_pred_list, total_deno_pred_list 이결과를 가지고 개수를 파악
    #     # regression_model = Regression_Model(*args)
    #     # regression_model.estimate_the_number_of_spins(A_lists)
    #     # regression_model.estimate_specific_AB_values(A_lists_with_the_number_of_spins)
    #
    #     N_PULSE = 256
    #     EXISTING_SPINS = 1
    #     B_init, B_final = 2000, 15000 #### *** 단 여기는 N256에서 A가10보다 작을 때는 B_init = 1500으로 해야함!!
    #     model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)
    #     args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final,
    #             A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
    #     total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
    #     # total_raw_pred_list, total_deno_pred_list 이결과를 가지고 개수를 파악
    #     # regression_model = Regression_Model(*args)
    #     # regression_model.estimate_the_number_of_spins(total_raw_pred_list, total_deno_pred_list)
    #     # regression_model.estimate_specific_AB_values()
    #     print('Total computational time:', time.time() - tic)
    #
    #     hpc_model.close_pool()
    #     regression_model.close_pool()
    #
    # else:
    #     model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)
    #     if N_PULSE==32:
    #         args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final,
    #                 A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
    #     elif N_PULSE==256:
    #         args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final,
    #                 A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
    #     hpc_model = HPC_Model(*args)
    #     total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
    #
    #     print('Total computational time:', time.time() - tic)
    #
    #     predicted_periods = return_filtered_A_lists_wrt_pred(np.array(total_deno_pred_list[1,:]), np.array(total_A_lists), 0.8)
    #     hpc_model.close_pool()
    #
    #     np.save(SAVE_DIR_NAME + 'predicted_periods.npy', np.array(predicted_periods, dtype=object))
    #     print("✅ HPC 결과를 저장했습니다.")
    #     predicted_periods = np.load(SAVE_DIR_NAME + 'predicted_periods.npy', allow_pickle=True)
    #
    #     def split_A_range(A_init, A_final, step=50, group_size = 20):
    #         A_values = list(range(A_init, A_final, step))
    #         grouped = [A_values[i:i+group_size] for i in range(0, len(A_values), group_size)]
    #         return grouped
    #     predicted_periods = split_A_range(A_init, A_final)
    #
    #     # 디버깅
    #     print(f"predicted_periods: {predicted_periods}")
    #     print(f"len(predicted_periods): {len(predicted_periods)}")
    #
    #     zero_scale = 0.05
    #     regression_args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final,
    #                         A_step, A_range, B_init, B_final, zero_scale, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN)
    #     regression_model = Regression_Model(*regression_args)
    #     regression_results = regression_model.estimate_specific_AB_values(predicted_periods)
    #     regression_model.close_pool()
    #     np.save(SAVE_DIR_NAME + 'regression_results.npy', np.array(regression_results, dtype=object))
    #
    #     regression_results= np.load(SAVE_DIR_NAME + "regression_results.npy", allow_pickle=True)
    #     for i in range(len(regression_results)):
    #         print(f"hier_indices: {regression_results[i][1]}")
    #     # predicted_periods와 핵스핀 개수로 A, B 파라미터 예측 필요
    #
    #     # B 값이 15000 이상인 하이퍼핀 파라미터만 필터링해서 저장
    #     filtered_results = [res for res in regression_results if res[1] > 15000]  # res = [A, B] 형태라고 가정
    #     filtered_results = np.array(filtered_results)
    #
    #     save_path = SAVE_DIR_NAME + 'predicted_results_N32_B15000above.npy'
    #     np.save(save_path, filtered_results)
    #     print(f"✅ B > 15000인 결과를 {save_path} 에 저장했습니다.")

