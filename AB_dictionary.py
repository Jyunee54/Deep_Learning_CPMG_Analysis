import numpy as np
from tqdm import tqdm
from imports.utils import *

# ✅ 기본 파라미터 설정
A_range = np.arange(-50000, 50001, 250)  # 예: A = -50000 ~ 50000, step=250
B_range = np.arange(500, 20001, 500)     # 예: B = 500 ~ 20000, step=500
time_data = np.linspace(0, 3000, 300)    # 시간 축 (단위: us)
WL = 1                                   # Weighting parameter (예: 1)
PULSE = 32                               # CPMG pulse 개수

# ✅ AB → M 딕셔너리 생성
AB_dict = {}
for A in tqdm(A_range, desc="Generating AB dictionary"):
    for B in B_range:
        AB_key = (A, B)
        M_arr = M_list_return(time_data * 1e-6, WL, np.array([A, B]) * 2 * np.pi, PULSE)
        AB_dict[AB_key] = M_arr.astype(np.float32)

# ✅ 저장
save_path = './data/AB_target_dic/AB_target_dic_v4.npy'
np.save(save_path, AB_dict, allow_pickle=True)
print(f"✅ AB dictionary saved to: {save_path}")
