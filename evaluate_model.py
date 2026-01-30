"""
Evaluation Script
Loads pretrained models (best reward, best lateness) and runs a single simulation episode
to generate Gantt charts and KPIs for comparison.
"""

import config as cfg
import rl_agent
import main_RL
import reporting
from reporting import LOG
import os
import random
import numpy as np
import torch

# Explicitly set seeds for evaluation to ensure consistency with training
# Ideally, this should match the seed used during training
random.seed(cfg.RND_SEED)
np.random.seed(cfg.RND_SEED)
torch.manual_seed(cfg.RND_SEED)

# Set the directory containing the models you want to evaluate
EVAL_TARGET_DIR = os.path.join("Results_TH", "FIFO_RL_2026-01-18 01:17:33")

def evaluate_and_plot(model_filename, run_name):
    # Construct path to the model file
    model_path = os.path.join(EVAL_TARGET_DIR, model_filename)
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return

    # Temporarily update output directory so results are saved in the same folder
    original_output_dir = cfg.OUTPUT_DIR
    cfg.OUTPUT_DIR = EVAL_TARGET_DIR

    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {run_name} ({model_filename})")
    print(f"{'='*60}")

    # 1. Reset Global Logs
    LOG.amr_events = []
    LOG.flight_events = []
    LOG.state_snapshots = []

    # 2. Setup Agent & Load Model
    agent = rl_agent.get_charging_agent()
    # Ensure agent structure matches training (re-init if needed, but get_charging_agent handles singleton)
    agent.load_model(model_path)
    
    # Disable exploration for evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0 

    # 3. Load Data
    df_flights, t_last_arrival = main_RL.parse_arrivals(cfg.ARRIVAL_CSV, cfg.NUM_FLIGHTS)

    # 4. Run Simulation (Episode 999 for evaluation)
    # train_mode=False avoids storing transitions or training
    # return_artifacts=True to get kpi and fleets for logging
    episode_reward, num_decisions, flight_lateness, kpi, fleets, sim_duration = main_RL.run_episode(
        episode_num=999,
        agent=agent,
        df_flights=df_flights,
        t_last_arrival=t_last_arrival,
        train_mode=False,
        return_artifacts=True
    )

    print(f"  > Episode Reward: {episode_reward:.2f}")
    print(f"  > Total Flight Lateness: {flight_lateness:.2f} min")

    # 5. Export logs (Events & KPI)
    reporting._export_logs(kpi, fleets, sim_duration)
    
    # Rename logs to include run_name to prevent overwriting
    log_files = ["log_amr_events.csv", "log_flight_events.csv", "kpi_amr_utilization.csv"]
    for log_file in log_files:
        src = os.path.join(cfg.OUTPUT_DIR, log_file)
        dst = os.path.join(cfg.OUTPUT_DIR, log_file.replace(".csv", f"_{run_name}.csv"))
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
            print(f"  > Saved log: {dst}")

    # 6. Plot Gantt Chart with specific timestamp/suffix
    # The reporting module's _plot_gate_gantt uses cfg.OUTPUT_DIR and global LOG
    # We pass a custom suffix to differentiate the output file
    # For evaluate_model, we want cleaner names like plot_gate_gantt_best_reward.png
    plot_suffix = f"_{run_name}" 
    reporting._plot_gate_gantt(plot_suffix)
    
    # Restore epsilon (though script ends anyway)
    agent.epsilon = original_epsilon
    
    # Restore configuration
    cfg.OUTPUT_DIR = original_output_dir

if __name__ == "__main__":
    # Ensure RL charging is ON
    cfg.USE_RL_CHARGING = True
    
    # 1. Evaluate "Best Reward" Model
    evaluate_and_plot("best_model_reward.pth", "best_reward")

    # 2. Evaluate "Best Lateness" Model
    evaluate_and_plot("best_model_lateness.pth", "best_lateness")
    
    # 3. (Optional) Evaluate "Final" Model if exists
    # evaluate_and_plot("final_model.pth", "final_model")
