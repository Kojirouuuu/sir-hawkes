"""
Hawkes過程のシミュレーションをローカル環境で実行するスクリプト

このスクリプトは、ネットワーク上でHawkes過程のシミュレーションを
並列実行し、結果をファイルに保存します。
"""

import argparse
import os
import sys
import logging
import json
import time
import platform
import socket
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psutil
from multiprocessing import cpu_count
from dotenv import load_dotenv

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.hawkes import simulate_hawkes_with_network, kernel_exp, HawkesParams
from simulation.network import ER, BA, Network


def detect_environment():
    """
    実行環境を判定する関数
    
    Returns:
        dict: 環境情報を含む辞書
    """
    env_info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'hostname': socket.gethostname(),
        'cpu_count': cpu_count(),
        'memory_total': psutil.virtual_memory().total
    }
    
    # EC2環境の判定
    is_ec2 = False
    ec2_instance_type = None
    
    # EC2のメタデータエンドポイントにアクセスして判定
    try:
        import urllib.request
        import urllib.error
        
        # EC2のメタデータエンドポイントにアクセス
        req = urllib.request.Request('http://169.254.169.254/latest/meta-data/instance-type')
        req.timeout = 1  # 1秒でタイムアウト
        response = urllib.request.urlopen(req)
        ec2_instance_type = response.read().decode('utf-8')
        is_ec2 = True
    except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout):
        # ローカル環境の場合
        is_ec2 = False
    
    env_info['is_ec2'] = is_ec2
    env_info['ec2_instance_type'] = ec2_instance_type
    
    # 環境タイプの文字列表現
    if is_ec2:
        env_info['environment_type'] = f"EC2-{ec2_instance_type}" if ec2_instance_type else "EC2"
    else:
        env_info['environment_type'] = f"Local-{platform.system()}"
    
    return env_info


# ─── Configuration ──────────────────────────────────────────────────────────────
@dataclass
class SimConfig:
    """シミュレーション設定を管理するデータクラス"""
    
    # ネットワーク設定
    nodes: int                    # ノード数
    avg_degree: float            # 平均次数
    initial_infected: int        # 初期感染ノード数
    t_max: int                  # シミュレーション終了時刻
    repeats: int                # 繰り返し回数
    r_frac_values: np.ndarray   # ２種類の分岐比の比率
    
    # パラメータ設定（デフォルト値あり）
    network: Network | None = None  # ネットワーク（Noneの場合は自動生成）
    q: float = 0                # 早期ノードの割合
    r_high: float = 1           # 高分岐比の値
    output_dir: Path = Path("./data")  # 出力ディレクトリ
    workers: int = None          # ワーカー数（None→自動設定）
    
    # 実行環境情報
    environment_info: dict = field(default_factory=detect_environment)


def init_worker_logging():
    """ワーカープロセスのログ設定を初期化"""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


# ─── Worker Function ────────────────────────────────────────────────────────────
def run_simulation(worker_id: int, cfg: SimConfig):
    """
    ワーカーごとにネットワークを生成し、各 r_frac 値・リピートでシミュレーションを実行
    
    Args:
        worker_id: ワーカーID
        cfg: シミュレーション設定
    """
    if worker_id == 0:
        logging.info(f"[W{worker_id}] シミュレーション開始")
    
    # ネットワーク生成
    if cfg.network is not None:
        g = cfg.network
    else:
        g = ER(cfg.nodes, cfg.avg_degree / (cfg.nodes - 1))
        # g = BA(cfg.nodes, cfg.avg_degree)  # BAネットワークを使用する場合
    
    # メモリ使用量をログ出力
    mem = psutil.Process().memory_info()
    if worker_id == 0:
        logging.debug(f"[W{worker_id}] メモリ使用量 - VM: {mem.vms/1e6:.1f}MB, RSS: {mem.rss/1e6:.1f}MB")

    # 結果格納用配列を初期化
    nr = len(cfg.r_frac_values)
    times_all = np.empty((nr, cfg.repeats), dtype=object)
    nodes_all = np.empty((nr, cfg.repeats), dtype=object)

    # 早期ノード数を計算
    early_num = int(cfg.nodes * cfg.q)

    # 各分岐比r_fracについてシミュレーション実行
    for i, r_frac in enumerate(cfg.r_frac_values):
        # 進捗表示（10回に1回または最後の回）
        visible = i % 10 == 0 or i == len(cfg.r_frac_values) - 1
        if visible and worker_id == 0:
            logging.info(f"[W{worker_id}] 分岐比比率 r_frac = {r_frac:.3f} ({i+1}/{nr})")
        
        # 各リピートについてシミュレーション実行
        for r in range(cfg.repeats):
            # ノードをランダムに並び替えて早期・後期ノードを決定
            nodes = np.arange(cfg.nodes)
            np.random.shuffle(nodes)  # numpy配列のshuffleはnp.random.shuffleを使用
            late_nodes = nodes[early_num:]
            
            # 分岐比を設定
            r_values = np.ones(cfg.nodes) * cfg.r_high
            r_values[late_nodes] = cfg.r_high * r_frac
            
            # Hawkes過程のパラメータを設定
            params = HawkesParams(
                kernel=kernel_exp, 
                dim=cfg.nodes, 
                r=r_values,  # 各ノードの分岐比
                b=1.0, 
                K=0, 
                c=0, 
                theta=0
            )
            
            # シミュレーション実行
            t, n = simulate_hawkes_with_network(
                t_max=cfg.t_max,
                initial_infected_num=cfg.initial_infected,
                p=params,
                g=g
            )
            
            # 結果を保存
            times_all[i, r] = t
            nodes_all[i, r] = n

    # 結果をファイルに保存
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    
    # シミュレーション結果を保存
    np.savez_compressed(
        out / f"{worker_id:02d}_results.npz",
        event_times=times_all, 
        event_nodes=nodes_all
    )
    
    # ネットワーク情報を保存（ネットワークが自動生成された場合のみ）
    if cfg.network is None:
        np.savez_compressed(
            out / f"{worker_id:02d}_network.npz",
            N=g.N, 
            k_ave=g.k_ave,
            edge_list=g.edge_list,
            address_list=g.address_list,
            cursor_list=g.cursor_list
        )
    
    logging.info(f"[W{worker_id}] シミュレーション完了")


def save_params(cfg: SimConfig):
    """
    シミュレーション設定をファイルに保存
    
    Args:
        cfg: シミュレーション設定
    """
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # JSONに保存するための設定辞書を作成（r_frac_valuesは除外）
    config_dict = cfg.__dict__.copy()
    config_dict.pop('r_frac_values', None)  # r_frac_valuesを除外
    if cfg.network is not None:
        config_dict.pop('network', None)        # networkオブジェクトも除外
        config_dict['is_network_common'] = True
    else:
        config_dict['is_network_common'] = False

    # Pathオブジェクトを文字列に変換
    if 'output_dir' in config_dict:
        config_dict['output_dir'] = str(config_dict['output_dir'])

    # 設定をJSONファイルに保存
    config_file_name = "config.json"
    with (out / config_file_name).open("w") as f:
        json.dump(config_dict, f, indent=2)

    # 分岐比の値をnumpy配列として保存
    np.save(out / "r_frac_values.npy", cfg.r_frac_values)
    
    # 環境情報をログ出力
    env_info = cfg.environment_info
    logging.info(f"実行環境: {env_info['environment_type']}")
    logging.info(f"ホスト名: {env_info['hostname']}")
    logging.info(f"CPU数: {env_info['cpu_count']}")
    logging.info(f"メモリ: {env_info['memory_total'] / 1e9:.1f}GB")
    if env_info['is_ec2'] and env_info['ec2_instance_type']:
        logging.info(f"EC2インスタンスタイプ: {env_info['ec2_instance_type']}")
    
    logging.info(f"設定を保存しました: {out}")


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    """メイン関数"""
    # 環境変数を読み込み
    load_dotenv()

    # Parameters
    nodes = 100
    avg_degree = 10.0
    initial_infected = 1
    t_max = 100
    repeats = 80

    r_frac_min = 0.1
    r_frac_max = 0.2
    r_frac_num = 100

    q = 0.02
    r_high = 0.5

    output_dir = Path(f"./data_ER_N={nodes}_r_high={r_high}_q={q}_exp")
    
    # 出力ディレクトリが存在しない場合は作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # network = None
    network = ER(nodes, avg_degree / (nodes - 1))
    np.savez_compressed(
        output_dir / "common_network.npz",
        N=network.N, 
        k_ave=network.k_ave,
        edge_list=network.edge_list,
        address_list=network.address_list,
        cursor_list=network.cursor_list
        )

    # シミュレーション設定
    cfg = SimConfig(
        nodes=nodes,                    # ノード数
        avg_degree=avg_degree,          # 平均次数
        network=network,                # ネットワーク
        initial_infected=int(initial_infected),  # 初期感染ノード数
        t_max=t_max,                    # シミュレーション終了時刻
        repeats=repeats,                # 繰り返し回数
        output_dir=output_dir,  # 出力ディレクトリ
        workers=cpu_count(),            # ワーカー数（CPU数に設定）
        r_frac_values=np.linspace(r_frac_min, r_frac_max, r_frac_num),  # 分岐比の範囲
        q=q,
        r_high=r_high
    )

    # 設定をファイルに保存
    save_params(cfg)

    # ログ設定
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    
    logging.info("=== Hawkes過程シミュレーション開始 ===")
    logging.info(f"実行環境: {cfg.environment_info['environment_type']}")
    logging.info(f"設定: ノード数={cfg.nodes}, 平均次数={cfg.avg_degree}, "
                f"初期感染={cfg.initial_infected}, 終了時刻={cfg.t_max}")
    logging.info(f"並列処理: {cfg.workers}ワーカー")
    
    start = time.time()

    # プロセスプールで並列実行
    with ProcessPoolExecutor(
        max_workers=cfg.workers, 
        initializer=init_worker_logging
    ) as executor:
        
        # 各ワーカーにタスクを割り当て
        futures = {
            executor.submit(run_simulation, wid, cfg): wid
            for wid in range(cfg.workers)
        }
        
        # 完了したタスクを処理
        for future in as_completed(futures):
            wid = futures[future]
            try:
                future.result()
                logging.info(f"[W{wid}] 正常終了")
            except Exception as e:
                logging.error(f"[W{wid}] エラー発生: {e}")

    # 実行時間を計算・表示
    elapsed = time.time() - start
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logging.info("=== シミュレーション完了 ===")
    logging.info(f"総実行時間: {int(hours)}時間 {int(minutes)}分 {int(seconds)}秒")


if __name__ == "__main__":
    main()
