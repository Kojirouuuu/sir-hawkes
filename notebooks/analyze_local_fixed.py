import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from multiprocessing import cpu_count

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath('.'))
sys.path.append(current_dir)
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from simulation.hawkes import HawkesParams, kernel, thinning_rate, lamb_i
from simulation.network import ER, Network

# グラフ全体のフォント設定
plt.rcParams['font.family'] = 'serif'  # 全体のフォントをSerifに設定
plt.rcParams['mathtext.fontset'] = 'cm'  # 数式のフォントをComputer Modernに設定
plt.rcParams['mathtext.rm'] = 'serif'  # TeXの通常フォントをSerifに設定
plt.rcParams['font.size'] = 18  # デフォルトフォントサイズ
plt.rcParams['axes.labelsize'] = 18  # 軸ラベルのフォントサイズ
plt.rcParams['axes.titlesize'] = 22  # タイトルのフォントサイズ
plt.rcParams['legend.fontsize'] = 16  # 凡例のフォントサイズ
plt.rcParams['grid.color'] = 'gray'  # グリッドの色を薄い灰色に設定
plt.rcParams['grid.linestyle'] = ':'  # グリッドを点線に設定
plt.rcParams['grid.linewidth'] = 0.5  # グリッドの線幅を設定

# パラメータ設定
N = 100

# データ読み込み
event_data = np.load(os.path.join(current_dir, f"data_N={N}", "03_results.npz"), allow_pickle=True)
network_data = np.load(os.path.join(current_dir, f"data_N={N}", "03_network.npz"), allow_pickle=True)

event_times = event_data["event_times"]
event_nodes = event_data["event_nodes"]

print(f"event_times.shape: {event_times.shape}")
print(f"event_nodes.shape: {event_nodes.shape}")
print(f"event_times[0, 0]: {event_times[0, 0]}")
print(f"event_nodes[0, 0]: {event_nodes[0, 0]}")

# ネットワーク設定
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

N = network_data["N"]
k_ave = network_data["k_ave"]
edge_list = network_data["edge_list"]
address_list = network_data["address_list"]
cursor_list = network_data["cursor_list"]
network = Network(N, k_ave, edge_list, address_list, cursor_list)
a_values = np.load(os.path.join(current_dir, f"data_N={N}", "a_values.npy"))

# プロット1: 強度関数の可視化
t = np.arange(0, 20, 0.001)
a_idx_list = [10, 50, 90]
which_repeat = 2

iidx_list = np.random.choice(range(N), size=5, replace=False)

fig, axes = plt.subplots(
    nrows=2,
    ncols=len(a_idx_list),
    figsize=(6 * len(a_idx_list), 8),
)

for idx, a_idx in enumerate(a_idx_list):
    a = a_values[a_idx]
    params = HawkesParams(kernel=kernel, dim=N, a=a, b=1)
    event_times_cur = event_data["event_times"][a_idx, which_repeat]
    event_nodes_cur = event_data["event_nodes"][a_idx, which_repeat]

    for iidx, i in enumerate(iidx_list):
        lam_i_t = lamb_i(i=i, t=t, event_times=event_times_cur, event_nodes=event_nodes_cur, p=params, g=network)
        axes[0, idx].plot(t, lam_i_t, color=colors[iidx], label=rf"$i = {i}$")
    
    axes[0, idx].set_xlabel("t")
    axes[0, idx].set_ylabel(rf"$\lambda_i(t|H_t)$")
    axes[0, idx].set_title(rf"$a = {a}$")
    axes[0, idx].grid(linestyle=":")
    axes[0, idx].legend()

    lam_t = np.array([lamb_i(i, t, event_times_cur, event_nodes_cur, params, network) for i in range(network.N)]).sum(axis=0)
    axes[1, idx].plot(t, lam_t, color=colors[-1])
    axes[1, idx].set_ylabel(rf"$\lambda(t|H_t)$")
    axes[1, idx].grid(linestyle=":")

fig.tight_layout()
plt.show()

# イベント数の確認
cur_event_data = np.load(os.path.join(current_dir, f"data_N={N}", f"05_results.npz"), allow_pickle=True)

idxa = 90
a = a_values[idxa]
idxw_list = np.arange(0, 40, 2)
print(f"a: {a}")
for idxw in idxw_list:
    print(f"idxw: {idxw}")
    print(f"event_nodes: {cur_event_data['event_nodes'][idxa, idxw]}")
    print(f"event_times: {cur_event_data['event_times'][idxa, idxw]}")
    print(f"event_num: {cur_event_data['event_times'][idxa, idxw].shape[0]}")
    print()

# イベント数のヒストグラム
repeats = event_data["event_times"].shape[1]
num_workers = cpu_count()
a_idx_list = [10, 30, 50, 70, 90]
event_num = np.zeros((len(a_idx_list), repeats * num_workers))

for idx, a_idx in enumerate(a_idx_list):
    a = a_values[a_idx]

    for repeat_idx in range(repeats):
        for worker_id in range(cpu_count()):
            cur_event_data = np.load(os.path.join(current_dir, f"data_N={N}", f"{worker_id:02d}_results.npz"), allow_pickle=True)
            event_num[idx, repeat_idx * num_workers + worker_id] = cur_event_data["event_times"][a_idx, repeat_idx].shape[0]

fig, axes = plt.subplots(
    nrows=1,
    ncols=len(a_idx_list),
    figsize=(4 * len(a_idx_list), 4),
)
for idx, a_idx in enumerate(a_idx_list):
    axes[idx].hist(event_num[idx], bins=20)
    axes[idx].set_xlabel("event_num")
    axes[idx].set_ylabel("count")
    axes[idx].set_title(rf"$a = {a_values[a_idx]:.4f}$")

fig.tight_layout()
plt.show()

# データ構造の確認と修正
print("Available keys in event_data:")
print(event_data.keys())

# 正しいデータ構造を使用
repeats = event_data["event_times"].shape[1]
len_args = event_data["event_times"].shape[0]

shape = (len_args, repeats * cpu_count())
event_times_all = np.zeros(shape, dtype=object)
event_nodes_all = np.zeros(shape, dtype=object)

for i in range(len_args):
    if i % 10 == 0 or i == len_args-1:
        print(f"i: {i}")
    for worker_id in range(cpu_count()):
        cur_event_data = np.load(os.path.join(current_dir, f"data_N={N}", f"{worker_id:02d}_results.npz"), allow_pickle=True)
        event_times_all[i, worker_id * repeats:(worker_id + 1) * repeats] = cur_event_data["event_times"][i]
        event_nodes_all[i, worker_id * repeats:(worker_id + 1) * repeats] = cur_event_data["event_nodes"][i]

print(event_times_all.shape)
print(event_nodes_all.shape)

# 強度関数の平均計算
N = network_data["N"]
k_ave = network_data["k_ave"]
edge_list = network_data["edge_list"]
address_list = network_data["address_list"]
cursor_list = network_data["cursor_list"]
network = Network(N, k_ave, edge_list, address_list, cursor_list)

t = np.linspace(0, 20, 100)
na = 50
a_values = np.linspace(0.08, 0.12, na)
lamb_ave_a = np.zeros((len(a_values), len(t)))

fig, axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(8, 5),
)

a_idx_list = [30, 35, 40]

for idx, a_idx in enumerate(a_idx_list):
    a = a_values[a_idx]
    is_print = True
    if is_print:
        print(f"a_idx: {a_idx}")
    for worker_id in range(cpu_count()):
        if is_print and worker_id%4 == 0:
            print(f" --> worker_id: {worker_id}")
        cur_network_data = np.load(os.path.join(current_dir, f"data_N={N}", f"{worker_id:02d}_network.npz"), allow_pickle=True)
        cur_network = Network(cur_network_data["N"], cur_network_data["k_ave"], cur_network_data["edge_list"], cur_network_data["address_list"], cur_network_data["cursor_list"])

        params = HawkesParams(kernel=kernel, dim=N, a=a, b=1)
        lam_t = np.zeros(len(t))
        for repeat_idx in range(repeats):
            lam_t += np.array([lamb_i(i, t, event_times_all[a_idx, worker_id * repeats + repeat_idx], event_nodes_all[a_idx, worker_id * repeats + repeat_idx], params, cur_network) for i in range(cur_network.N)]).sum(axis=0)

    lamb_ave_a[a_idx] = lam_t / (repeats * cpu_count())

for a_idx in a_idx_list:
    axes.plot(t, lamb_ave_a[a_idx], label=rf"$a = {a_values[a_idx]:.4f}$")
axes.set_xlabel("t")
axes.set_ylabel(rf"$\lambda(t|H_t)$")
axes.grid(linestyle=":")
axes.legend()
fig.tight_layout()
plt.show()

# イベント数の平均計算
event_num_ave = np.zeros(len(a_values))

for a_idx in range(len(a_values)):
    print(f"a_idx: {a_idx}")
    valid_num = 0
    for repeat_idx in range(repeats):
        for worker_id in range(cpu_count()):
            cur_event_data = np.load(os.path.join(current_dir, f"data_N={N}", f"{worker_id:02d}_results.npz"), allow_pickle=True)
            if cur_event_data["event_times"][a_idx, repeat_idx].shape[0] != 100001:
                event_num_ave[a_idx] += cur_event_data["event_times"][a_idx, repeat_idx].shape[0]
                valid_num += 1
    if valid_num > 0:
        event_num_ave[a_idx] /= valid_num

fig, axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(8, 5),
)
axes.plot(a_values, event_num_ave)
axes.set_xlabel("a")
axes.set_ylabel("event_num_ave")
plt.show()

# 最適化されたイベント数計算
import os
import numpy as np
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

#--- 1) データ読み込み・キャッシュ ---
data_dir = os.path.join(current_dir, f"data_N={N}")
n_workers = cpu_count()

# worker_data[i] = 各 worker の event_times 配列 (shape=(len(a_values), repeats) の object array)
worker_data = []
for wid in range(n_workers):
    path = os.path.join(data_dir, f"{wid:02d}_results.npz")
    with np.load(path, allow_pickle=True) as d:
        worker_data.append(d["event_times"])

#--- 2) 長さ行列の一括取得とマスク集計 ---
A = len(a_values)
# 合計長さ・有効カウントを a_idx ごとに保持
total_lengths = np.zeros(A, dtype=float)
valid_counts  = np.zeros(A, dtype=int)

for ev_times in worker_data:
    # ev_times は shape=(A, repeats) の object array
    # 各セルの shape[0] (長さ) をベクトル化して取得
    lengths = np.vectorize(lambda arr: arr.shape[0])(ev_times)  # shape=(A, repeats)
    mask = (lengths != 100001)                               # True なら集計対象
    
    # a_idx ごとに足し込み
    total_lengths += (lengths * mask).sum(axis=1)
    valid_counts  += mask.sum(axis=1)

#--- 3) 平均を計算 ---
event_num_ave = total_lengths / valid_counts

#--- 4) プロット ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(a_values, event_num_ave, marker='o')
ax.set_xlabel("a")
ax.set_ylabel("event_num_ave")
plt.tight_layout()
plt.show() 