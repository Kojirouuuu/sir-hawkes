"""
Hawkes過程のシミュレーションをローカル環境あるいは AWS EC2 で実行するスクリプト

- 並列にネットワーク付き Hawkes 過程を実行し、結果を .npz に保存
- 保存したファイルは S3 へアップロード
- 実行環境 (EC2 or Local) を高精度に判定し、ログおよび設定ファイルに残す
"""

# ─── 標準ライブラリ ────────────────────────────────────────────────
import json
import logging
import os
import platform
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# ─── サードパーティライブラリ ───────────────────────────────────────
import numpy as np
import psutil
from dotenv import load_dotenv

# ─── プロジェクト内モジュール ─────────────────────────────────────
# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.hawkes import simulate_hawkes_with_network, kernel_exp, HawkesParams
from simulation.network import ER, BA, Network
from s3.s3_io import upload_to_s3

# ─────────────────────────────────────────────────────────────────────
# 1. 実行環境判定ロジック
# ─────────────────────────────────────────────────────────────────────
IMDS_BASE = "http://169.254.169.254/latest"

@dataclass
class EnvironmentInfo:
    """実行環境に関するメタ情報"""

    is_ec2: bool
    environment_type: str      # "EC2-c6i.large" / "Local-Darwin" など
    detect_method: str         # "IMDSv2" / "IMDSv1" / "UUID" / "Unknown"
    platform: str
    platform_release: str
    machine: str
    processor: str
    hostname: str
    cpu_count: int
    memory_total: int          # bytes


def _try_imdsv2(path: str, timeout: float = 0.5) -> str | None:
    """IMDSv2 でメタデータを取得。取れなければ None"""
    token_req = urllib.request.Request(
        f"{IMDS_BASE}/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    try:
        token = urllib.request.urlopen(token_req, timeout=timeout).read().decode()
        md_req = urllib.request.Request(
            f"{IMDS_BASE}{path}",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(md_req, timeout=timeout).read().decode()
    except Exception:
        return None


def _try_imdsv1(path: str, timeout: float = 0.5) -> str | None:
    """IMDSv1 でメタデータを取得 (フォールバック用)。"""
    try:
        return urllib.request.urlopen(f"{IMDS_BASE}{path}", timeout=timeout).read().decode()
    except Exception:
        return None


def _uuid_looks_like_ec2() -> bool:
    """/sys 内の uuid が EC2 っぽいかを判定"""
    uuid_paths = [
        "/sys/hypervisor/uuid",
        "/sys/devices/virtual/dmi/id/product_uuid",
        "/sys/class/dmi/id/board_asset_tag",  # Nitro 系
    ]
    for p in uuid_paths:
        try:
            with open(p) as f:
                if f.read().strip().lower().startswith("ec2"):
                    return True
        except FileNotFoundError:
            continue
    return False


def detect_environment() -> EnvironmentInfo:
    """EC2 / ローカル判定を行い、EnvironmentInfo を返す"""
    common = dict(
        platform=platform.system(),
        platform_release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor(),
        hostname=socket.gethostname(),
        cpu_count=cpu_count(),
        memory_total=psutil.virtual_memory().total,
    )

    # A. IMDSv2
    instance_type = _try_imdsv2("/meta-data/instance-type")
    if instance_type:
        return EnvironmentInfo(
            is_ec2=True,
            environment_type=f"EC2-{instance_type}",
            detect_method="IMDSv2",
            **common,
        )

    # B. IMDSv1
    instance_type = _try_imdsv1("/meta-data/instance-type")
    if instance_type:
        return EnvironmentInfo(
            is_ec2=True,
            environment_type=f"EC2-{instance_type}",
            detect_method="IMDSv1",
            **common,
        )

    # C. UUID 判定
    if _uuid_looks_like_ec2():
        return EnvironmentInfo(
            is_ec2=True,
            environment_type="EC2-uuid",
            detect_method="UUID",
            **common,
        )

    # D. Local 環境
    return EnvironmentInfo(
        is_ec2=False,
        environment_type=f"Local-{platform.system()}",
        detect_method="Unknown",
        **common,
    )


# ─────────────────────────────────────────────────────────────────────
# 2. シミュレーション設定データクラス
# ─────────────────────────────────────────────────────────────────────
@dataclass
class SimConfig:
    """Hawkes シミュレーションの設定を保持するデータクラス"""

    # --- ネットワーク・シミュレーション関連 ---
    nodes: int
    avg_degree: float
    initial_infected: int
    t_max: int
    repeats: int
    r_frac_values: np.ndarray

    # --- オプション ---
    network: Network | None = None
    q: float = 0.0
    r_high: float = 1.0
    output_dir: Path = Path("./data")
    workers: int | None = None

    # --- 実行環境情報 ---
    environment_info: EnvironmentInfo = field(default_factory=detect_environment)

# ─────────────────────────────────────────────────────────────────────
# 3. 便利ユーティリティ
# ─────────────────────────────────────────────────────────────────────

def init_worker_logging():
    """ワーカープロセス用のログ設定"""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ─────────────────────────────────────────────────────────────────────
# 4. ワーカーで実行されるシミュレーション関数
# ─────────────────────────────────────────────────────────────────────

def run_simulation(worker_id: int, cfg: SimConfig):
    """各ワーカーが r_frac × repeats 分シミュレーションを実行"""
    if worker_id == 0:
        logging.info(f"[W{worker_id}] シミュレーション開始")

    # --- ネットワーク生成 (共有 or 個別) ---
    g = cfg.network if cfg.network is not None else ER(cfg.nodes, cfg.avg_degree / (cfg.nodes - 1))

    # --- メモリログ ---
    if worker_id == 0:
        mem = psutil.Process().memory_info()
        logging.debug(
            f"[W{worker_id}] メモリ使用量 - VM: {mem.vms/1e6:.1f}MB, RSS: {mem.rss/1e6:.1f}MB"
        )

    # --- 出力配列 ---
    nr = len(cfg.r_frac_values)
    times_all = np.empty((nr, cfg.repeats), dtype=object)
    nodes_all = np.empty((nr, cfg.repeats), dtype=object)

    # --- 早期・後期ノード分割 ---
    early_num = int(cfg.nodes * cfg.q)

    # --- r_frac ループ ---
    for i, r_frac in enumerate(cfg.r_frac_values):
        if (i % 10 == 0 or i == nr - 1) and worker_id == 0:
            logging.info(f"[W{worker_id}] r_frac = {r_frac:.3f} ({i+1}/{nr})")

        for r in range(cfg.repeats):
            nodes = np.arange(cfg.nodes)
            np.random.shuffle(nodes)
            late_nodes = nodes[early_num:]

            r_values = np.ones(cfg.nodes) * cfg.r_high
            r_values[late_nodes] = cfg.r_high * r_frac

            params = HawkesParams(
                kernel=kernel_exp,
                dim=cfg.nodes,
                r=r_values,
                b=1.0,
                K=0,
                c=0,
                theta=0,
            )

            t, n = simulate_hawkes_with_network(
                t_max=cfg.t_max,
                initial_infected_num=cfg.initial_infected,
                p=params,
                g=g,
            )

            times_all[i, r] = t
            nodes_all[i, r] = n

    # --- 保存 ---
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    results_file = cfg.output_dir / f"{worker_id:02d}_results.npz"
    np.savez_compressed(results_file, event_times=times_all, event_nodes=nodes_all)
    upload_to_s3(results_file, cfg.output_dir)

    if cfg.network is None:
        net_file = cfg.output_dir / f"{worker_id:02d}_network.npz"
        np.savez_compressed(
            net_file,
            N=g.N,
            k_ave=g.k_ave,
            edge_list=g.edge_list,
            address_list=g.address_list,
            cursor_list=g.cursor_list,
        )
        upload_to_s3(net_file, cfg.output_dir)

    logging.info(f"[W{worker_id}] シミュレーション完了")


# ─────────────────────────────────────────────────────────────────────
# 5. 設定ファイル保存
# ─────────────────────────────────────────────────────────────────────

def save_params(cfg: SimConfig):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON 出力 (r_frac_values / network オブジェクトを除外) ---
    config_dict = asdict(cfg)
    config_dict.pop("r_frac_values", None)
    if cfg.network is not None:
        config_dict.pop("network", None)
        config_dict["is_network_common"] = True
    else:
        config_dict["is_network_common"] = False

    config_dict["output_dir"] = str(cfg.output_dir)

    config_path = cfg.output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    upload_to_s3(config_path, cfg.output_dir)

    # --- r_frac_values を別ファイルで保存 ---
    np.save(cfg.output_dir / "r_frac_values.npy", cfg.r_frac_values)

    # --- ログ出力 ---
    env = cfg.environment_info
    logging.info(f"実行環境: {env.environment_type} ({env.detect_method})")
    logging.info(f"ホスト名: {env.hostname}")
    logging.info(f"CPU: {env.cpu_count} / メモリ: {env.memory_total/1e9:.1f}GB")
    if env.is_ec2 and "EC2-" in env.environment_type:
        logging.info(f"EC2 インスタンスタイプ: {env.environment_type.split('-', 1)[1]}")


# ─────────────────────────────────────────────────────────────────────
# 6. メイン関数
# ─────────────────────────────────────────────────────────────────────

def main():
    """エントリポイント"""
    load_dotenv()

    # --- パラメータ (CLI で上書き可能にするなら argparse を利用) ---
    nodes = 100
    avg_degree = 10.0
    initial_infected = 1
    t_max = 100
    repeats = 40

    r_frac_min = 0.02
    r_frac_max = 0.04
    r_frac_num = 100

    q = 0.02
    r_high = 2.0

    output_dir = Path(f"./data_ER_N={nodes}_r_high={r_high}_q={q}_exp")

    # --- ネットワークを共通で使う例 (None にすればワーカー毎に生成) ---
    network = ER(nodes, avg_degree / (nodes - 1))
    np.savez_compressed(
        output_dir / "common_network.npz",
        N=network.N,
        k_ave=network.k_ave,
        edge_list=network.edge_list,
        address_list=network.address_list,
        cursor_list=network.cursor_list,
    )

    cfg = SimConfig(
        nodes=nodes,
        avg_degree=avg_degree,
        network=network,
        initial_infected=initial_infected,
        t_max=t_max,
        repeats=repeats,
        r_frac_values=np.linspace(r_frac_min, r_frac_max, r_frac_num),
        q=q,
        r_high=r_high,
        output_dir=output_dir,
        workers=cpu_count(),
    )

    # --- ログ設定 ---
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.info("=== Hawkes 過程シミュレーション開始 ===")
    env = cfg.environment_info
    logging.info(f"実行環境: {env.environment_type} (検出方法: {env.detect_method})")
    logging.info(
        f"設定: N={cfg.nodes}, k_ave={cfg.avg_degree}, 初期感染={cfg.initial_infected}, t_max={cfg.t_max}"
    )
    logging.info(f"並列ワーカー: {cfg.workers}")

    # --- 設定を保存 ---
    save_params(cfg)

    start = time.time()

    with ProcessPoolExecutor(
        max_workers=cfg.workers,
        initializer=init_worker_logging,
    ) as executor:
        futures = {executor.submit(run_simulation, wid, cfg): wid for wid in range(cfg.workers)}
        for future in as_completed(futures):
            wid = futures[future]
            try:
                future.result()
                logging.info(f"[W{wid}] 正常終了")
            except Exception as e:
                logging.error(f"[W{wid}] エラー: {e}")

    # --- 終了ログ ---
    elapsed = time.time() - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    logging.info("=== シミュレーション完了 ===")
    logging.info(f"総実行時間: {int(h)}h {int(m)}m {int(s)}s")


if __name__ == "__main__":
    main()
