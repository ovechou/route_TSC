"""
批量测试脚本：用 20 个不同的 cav_seed 测试 AlphaRouter，
汇总 Travel Time / Delay / Waiting Time（CAV+HV 和 仅CAV），
输出每次结果 + min/max/mean 总表。
"""
import sys
import os
import csv
import numpy as np
import pandas as pd

os.environ['SUMO_HOME'] = "/usr/share/sumo"
sys.path.append(os.path.join(os.environ.get("SUMO_HOME"), 'tools'))

import torch
import traci

import config
import env
import lights
import episode
import tripinfo
from dqnagent import DQNAgent
from ppoagent import PPOAgent
from utils import Visualization
from arguments import get_common_args, get_ppo_arguments, get_mat_arguments, get_mappo_arguments, get_ma2c_arguments
import traffic_light

ALGO = "alpha_router"
MODEL_NAME = "0.5_alpha_router_0315"
MODEL_PATH = MODEL_NAME + "/best_model"
SUMOCFG = "./res/hangzhou/exe_2983.sumocfg"
RATIO = 0.5
CSV_CAV = 2983

SEEDS = [0, 4, 6, 7, 8, 10, 14, 16, 17, 18, 20, 24, 26, 27, 28, 30, 34, 36, 37, 38]


def run_one_test(cav_seed, sumo_seed='25'):
    torch.set_num_threads(1)

    ap = get_common_args(EXEC=True)
    args = ap.parse_args()
    if args.agent_type in {"PPO", "CenPPO"}:
        ap = get_ppo_arguments(EXEC=True)
    elif args.agent_type == "MAT":
        ap = get_mat_arguments(EXEC=True)
    elif args.agent_type == "MAPPO":
        ap = get_mappo_arguments(EXEC=True)
    elif args.agent_type == "MA2C":
        ap = get_ma2c_arguments(EXEC=True)
    args = ap.parse_args()
    config.SPATIAL["TYPE"] = args.spatial

    output_dir = f'./simudata/{MODEL_NAME}'
    os.makedirs(output_dir, exist_ok=True)
    tripinfo.set_output_dir(output_dir)

    agents = lights.init(args, True, SUMOCFG, seed=sumo_seed, output_dir=output_dir)
    args.rate = RATIO
    ap.add_argument("--dir", type=str, default=MODEL_NAME)

    RouterNet = None
    adj_edge = None
    if args.router:
        adj_edge = env.find_adj_edge(f'./res/{args.map}/4_Phase.net.xml')
        state_dim = args.agent_num * args.road_feature * 4 * 3 + args.cav_feature
        if ALGO == "alpha_router":
            from algo.alpha_router_agent import AlphaRouterWrapper
            RouterNet = AlphaRouterWrapper(state_dim=state_dim, action_dim=args.direction, args=args)
        elif config.ROUTER_RL["ALGO"] == "DQN":
            RouterNet = DQNAgent(state_dim=state_dim, action_dim=args.direction, args=args, tl_id="veh", net_type="",
                                 net_config=config.ROUTER_RL)
        RouterNet.load_model(MODEL_PATH)

    gen_cav = []
    with open(f'./res/hangzhou/{cav_seed}_cav_{CSV_CAV}_{RATIO}.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            processed_row = [int(cell) for cell in row if cell.strip()]
            gen_cav.extend(processed_row)
    args.veh_num = CSV_CAV

    o = Visualization()
    config.SIMULATION['EXECUTE'] = True

    env.init_context(agents.get_agent_list())
    traci.vehicletype.setTau('DEFAULT_VEHTYPE', 2.0)

    reward, t, c, s, l, w, d, rt, _ = episode.run(0, agents, o, args, True, MODEL_NAME, RouterNet, adj_edge, gen_cav)

    cav_set = set(gen_cav)
    all_duration = [float(x) for x in tripinfo.get_tripinfo_list('duration')]
    all_delay = [float(x) for x in tripinfo.get_tripinfo_list('timeLoss')]
    all_waiting = [float(x) for x in tripinfo.get_tripinfo_list('waitingTime')]

    cav_duration = [float(x) for x in tripinfo.get_tripinfo_list('duration', cav_set=cav_set)]
    cav_delay = [float(x) for x in tripinfo.get_tripinfo_list('timeLoss', cav_set=cav_set)]
    cav_waiting = [float(x) for x in tripinfo.get_tripinfo_list('waitingTime', cav_set=cav_set)]

    result = {
        'cav_seed': cav_seed,
        'TravelTime_ALL': np.mean(all_duration) if all_duration else 0,
        'TravelTime_CAV': np.mean(cav_duration) if cav_duration else 0,
        'Delay_ALL': np.mean(all_delay) if all_delay else 0,
        'Delay_CAV': np.mean(cav_delay) if cav_delay else 0,
        'WaitingTime_ALL': np.mean(all_waiting) if all_waiting else 0,
        'WaitingTime_CAV': np.mean(cav_waiting) if cav_waiting else 0,
    }

    env.close()
    return result


if __name__ == "__main__":
    sys.argv = ["batch_test.py", "--algo", ALGO]

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"  Test {i+1}/20 | cav_seed={seed}")
        print(f"{'='*60}")
        try:
            r = run_one_test(seed)
            results.append(r)
            print(f"  => TravelTime(ALL)={r['TravelTime_ALL']:.2f}, TravelTime(CAV)={r['TravelTime_CAV']:.2f}")
            print(f"     Delay(ALL)={r['Delay_ALL']:.2f}, Delay(CAV)={r['Delay_CAV']:.2f}")
            print(f"     Waiting(ALL)={r['WaitingTime_ALL']:.2f}, Waiting(CAV)={r['WaitingTime_CAV']:.2f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("没有成功的测试结果")
        sys.exit(1)

    df = pd.DataFrame(results)

    summary = pd.DataFrame({
        'cav_seed': ['MIN', 'MAX', 'MEAN'],
        'TravelTime_ALL': [df['TravelTime_ALL'].min(), df['TravelTime_ALL'].max(), df['TravelTime_ALL'].mean()],
        'TravelTime_CAV': [df['TravelTime_CAV'].min(), df['TravelTime_CAV'].max(), df['TravelTime_CAV'].mean()],
        'Delay_ALL': [df['Delay_ALL'].min(), df['Delay_ALL'].max(), df['Delay_ALL'].mean()],
        'Delay_CAV': [df['Delay_CAV'].min(), df['Delay_CAV'].max(), df['Delay_CAV'].mean()],
        'WaitingTime_ALL': [df['WaitingTime_ALL'].min(), df['WaitingTime_ALL'].max(), df['WaitingTime_ALL'].mean()],
        'WaitingTime_CAV': [df['WaitingTime_CAV'].min(), df['WaitingTime_CAV'].max(), df['WaitingTime_CAV'].mean()],
    })

    full_table = pd.concat([df, summary], ignore_index=True)

    save_path = f'./simudata/{MODEL_NAME}/batch_test_summary.csv'
    full_table.to_csv(save_path, index=False, float_format='%.2f')

    print(f"\n\n{'='*80}")
    print(f"  AlphaRouter Batch Test Summary ({len(results)} seeds)")
    print(f"{'='*80}")
    print(f"\n{'Method':<12} {'TravelTime(s)':<30} {'Delay(s)':<30} {'WaitingTime(s)':<30}")
    print(f"{'':12} {'CAV+HV':<15} {'CAV':<15} {'CAV+HV':<15} {'CAV':<15} {'CAV+HV':<15} {'CAV':<15}")
    print("-" * 102)

    for _, row in df.iterrows():
        print(f"seed={int(row['cav_seed']):<6} {row['TravelTime_ALL']:<15.2f} {row['TravelTime_CAV']:<15.2f} "
              f"{row['Delay_ALL']:<15.2f} {row['Delay_CAV']:<15.2f} "
              f"{row['WaitingTime_ALL']:<15.2f} {row['WaitingTime_CAV']:<15.2f}")

    print("-" * 102)
    for _, row in summary.iterrows():
        print(f"{row['cav_seed']:<12} {row['TravelTime_ALL']:<15.2f} {row['TravelTime_CAV']:<15.2f} "
              f"{row['Delay_ALL']:<15.2f} {row['Delay_CAV']:<15.2f} "
              f"{row['WaitingTime_ALL']:<15.2f} {row['WaitingTime_CAV']:<15.2f}")

    print(f"\n结果已保存到: {save_path}")
