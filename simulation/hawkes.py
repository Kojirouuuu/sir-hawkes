from .network import Network
import numpy as np
from typing import Callable, List
from scipy.special import gamma


class HawkesParams:
    """Hawkes過程のパラメータを管理するクラス"""
    
    def __init__(self,
                 kernel: Callable[[np.ndarray], np.ndarray],
                 dim: int,
                 r: np.ndarray, 
                 b: float,
                 K: float, 
                 c: float, 
                 theta: float,
                 rho: np.ndarray = None):
        """
        Args:
            kernel: カーネル関数
            dim: 次元数
            r: イベントiの分岐比
            b: 指数減衰パラメータ
            K: 冪函数カーネルの強度パラメータ
            c: 冪函数カーネルのオフセットパラメータ
            theta: 冪函数カーネルの指数パラメータ
            rho: 基底強度（デフォルト: 零ベクトル）
        """
        self.kernel = kernel
        self.dim = dim
        self.r = r
        self.b = b
        self.K = K
        self.c = c
        self.theta = theta
        self.rho = np.zeros(dim) if rho is None else rho


def kernel_exp(tau: np.ndarray, p: HawkesParams, event: int) -> np.ndarray:
    """
    指数関数カーネル: g(t) = a·b·exp(-b·t)·1_{t>=0}
    
    Args:
        tau: 時間差の配列
        p: HawkesParamsオブジェクト
        
    Returns:
        カーネル値の配列
    """
    r = p.r
    b = p.b
    if event == -1:
        r = np.zeros(1)
    else:
        r = r[event]

    return r * b * np.exp(-b * tau) * (tau >= 0)

# TODO: 冪函数カーネルを実装したらここを有効にする
# def kernel_pow(tau: np.ndarray, p: HawkesParams, event: int) -> np.ndarray:
#     """
#     冪函数カーネル: g(t) = K/(t+c)^{1+θ}·1_{t>=0}
    
#     Args:
#         tau: 時間差の配列
#         p: HawkesParamsオブジェクト
        
#     Returns:
#         カーネル値の配列
#     """
#     K = p.K
#     c = p.c
#     theta = p.theta
#     return K * (tau + c)**(-(1 + theta)) * (tau >= 0) * p.r[event]


def lamb_i(i: int, 
           t: np.ndarray, 
           event_times: List[float], 
           event_nodes: List[int], 
           p: HawkesParams, 
           g: Network) -> np.ndarray:
    """
    イベントiの条件付き強度関数
    
    Args:
        i: ノードインデックス
        t: 時間配列
        event_times: イベント時刻のリスト
        event_nodes: イベントノードのリスト
        p: HawkesParamsオブジェクト
        g: Networkオブジェクト
        
    Returns:
        イベントiの条件付き強度の配列
    """
    # ノードiの隣接ノード数を取得
    dim_i = g.cursor_list[i] - g.address_list[i]
    dt = np.zeros((dim_i, len(event_times)))
    sigma_j = np.zeros((dim_i, len(t)))

    # 各隣接ノードjについて影響を計算
    for jidx in range(dim_i):
        j = g.edge_list[g.address_list[i] + jidx]
        
        # ノードjのイベント履歴を収集
        history = []
        for eidx, et in enumerate(event_times):
            if event_nodes[eidx] == j:
                history.append(et)
        
        # 時間差を計算
        dt = t[:, None] - np.array(history)[None, :]
        
        # カーネル関数による影響を計算
        sigma_j[jidx, :] = (p.kernel(dt, p, j)).sum()

    return p.rho[i] + sigma_j.sum(axis=0)


def thinning_rate(lamb_star: np.ndarray, 
                  elapsed: float, 
                  p: HawkesParams, 
                  g: Network) -> np.ndarray:
    """
    時間経過後の強度関数を計算
    
    Args:
        lamb_star: 現在の強度
        elapsed: 経過時間
        p: HawkesParamsオブジェクト
        g: Networkオブジェクト
        
    Returns:
        時間経過後の強度配列
    """
    if p.kernel is kernel_exp:
        return p.rho + (lamb_star - p.rho) * np.exp(-p.b * elapsed)
    
    # TODO: 冪函数カーネルを実装したらここを有効にする
    # elif p.kernel is kernel_pow:
    #     return p.rho + (lamb_star - p.rho) * (elapsed + p.c)**(-(1 + p.theta))
    else:
        raise ValueError("Invalid kernel")


def simulate_hawkes_with_network(t_max: float, 
                                initial_infected_num: int, 
                                p: HawkesParams, 
                                g: Network, 
                                max_events: int = 100000) -> np.ndarray:
    """
    Hawkes過程のシミュレーションを実行
    
    Args:
        t_max: シミュレーション終了時刻
        initial_infected_num: 初期感染ノード数
        p: HawkesParamsオブジェクト
        g: Networkオブジェクト
        max_events: 最大イベント数
        
    Returns:
        (イベント時刻配列, イベントノード配列)のタプル
    """
    t_star = 0.0
    event_times: List[float] = []
    event_nodes: List[int] = []

    # 初期感染ノードをランダムに選択
    initial_infected = np.random.choice(g.N, initial_infected_num, replace=False)
    lamb_star = np.zeros(g.N)
    
    # 初期強度を設定
    for node in initial_infected:
        lamb_star[node] = p.kernel(tau = np.array([0]), p = p, event = -1)

        for neighbor in g.edge_list[g.address_list[node]:g.cursor_list[node]]:
            lamb_star[neighbor] += p.kernel(tau = np.array([0]), p = p, event = node)
    
    event_times = [0]
    event_nodes = initial_infected.tolist()

    # メインシミュレーションループ
    while t_star < t_max and len(event_times) < max_events:
        # 1) 次の候補時刻をサンプリング
        wait = np.random.exponential(1.0 / lamb_star.sum())
        t_star += wait
        
        if t_star > t_max:
            break

        # 2) 時間経過後の強度を計算
        current_intensity = thinning_rate(lamb_star, wait, p, g)
        
        # 3) イベント発生の判定（thinning法）
        accept_prob = current_intensity.sum() / lamb_star.sum()
        
        if accept_prob >= np.random.rand():
            # イベントが発生した場合
            event_times.append(t_star)

            # イベント発生ノードを選択
            probs = current_intensity.copy()
            probs = np.clip(probs, 0, None)
            total = probs.sum()
            
            if total <= 0:
                break
                
            probs = probs / total
            
            # 確率の合計を厳密に1にするための調整
            imax = np.argmax(probs)
            probs[imax] += 1.0 - probs.sum()

            # ノードを選択
            node = np.random.choice(p.dim, p=probs)
            event_nodes.append(int(node))
            
            # 強度を更新
            lamb_star = current_intensity
            for neighbor in g.edge_list[g.address_list[node]:g.cursor_list[node]]:
                lamb_star[neighbor] += p.kernel(tau = np.array([0]), p = p, event = node)
                
            if len(event_times) > max_events:
                break
        else:
            # イベントが発生しなかった場合
            lamb_star = current_intensity

    return np.array(event_times), np.array(event_nodes)
