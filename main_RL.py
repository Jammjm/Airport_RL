"""
Main RL Training Script - Synchronized with main_final.py simulation environment
Tracks ONLY flight lateness (AMR lateness is NOT included in reward)

Episode-based training loop that:
1. Initializes simulation (synchronized with main_final.py)
2. Steps through simulation until decision needed
3. Gets state from sim, selects action, returns to sim
4. Stores transitions and trains agent
5. Tracks flight lateness metrics as in main_final.py (for analysis/KPI only)
"""

import simpy
import pandas as pd
import numpy as np
from datetime import datetime
import random
from typing import Optional
import os
import sys
import matplotlib.pyplot as plt

import config as cfg
from routing import (
    NODE_POS, GATE_LABELS, _norm_label,
    _get_path_waypoints, _calculate_path_distance_and_time
)
from sim_model_RL import AMRFleet, ChargerBank, flight_starter
from reporting import KPIs, EventLogger, _setup_output_dir, _plot_gate_gantt, _export_logs, LOG
import rl_agent


def calculate_reward(
    state_vec: np.ndarray,
    action: int,
    t_completion: float,
    due_time: Optional[float] = None,
) -> float:
    """
    === REWARD LOGIC: UPDATED ===
    Implements the formula:
    r = - beta * max(0, t_completion - due_time) - gamma * I[SoC < tau and action==0] * (tau - SoC)
    
    Where:
    - t_completion is the time the job was completed (t_next in previous context)
    - due_time is the due time of the job just completed
    """
    tau = getattr(cfg, "MANDATORY_CHARGE_SOC", 0.2)
    gamma = getattr(cfg, "REWARD_GAMMA", 1.0)
    beta = 2.0 # Weight for tardiness penalty

    soc_before = float(state_vec[0]) if len(state_vec) else 0.0

    # 1. Tardiness Penalty
    tardiness = 0.0
    if due_time is not None:
        tardiness = max(0.0, t_completion - due_time)
    
    tardiness_penalty = beta * tardiness

    # 2. Mandatory Charge Penalty
    # INCREASED WEIGHT: To ensure cost of "Low Battery" > cost of "Charging Time"
    # If charging takes 30 mins, cost is ~60. 
    # We want (Tau - SoC) penalty to be > 60 when diff is small (e.g. 0.05).
    # New Multiplier: 2000.  (2000 * 0.05 = 100 > 60).
    mandatory_penalty = 0.0
    if soc_before < tau and action == 0:
        mandatory_penalty = 2000.0 * (tau - soc_before) 

    # Total Reward
    reward = -(tardiness_penalty + mandatory_penalty)
    
    # Scale down to keep Q/Loss magnitudes manageable
    return float(reward / 100.0)


def parse_arrivals(csv_path: str, num_flights: int) -> tuple[pd.DataFrame, float]:
    """Parse flight arrival data from CSV - synchronized with main_final.py"""
    df_raw = pd.read_csv(csv_path)
    df = df_raw.copy()
    df['ARR_TIME'] = pd.to_datetime(df['ARR_TIME'], errors='coerce')
    df['GATE'] = df['GATE'].astype(str).apply(_norm_label)

    # Filter for valid gates defined in map
    valid_gates = set(GATE_LABELS)
    gate_mask = df['GATE'].isin(valid_gates)
    time_mask = df['ARR_TIME'].notna()
    df = df[gate_mask & time_mask].copy()

    if len(df) == 0:
        raise ValueError(f"No valid flights remain after filtering for gates: {valid_gates}")

    # Sort by arrival time
    # df = df.sort_values('ARR_TIME').reset_index(drop=True)

    # Use first N flights if limited, otherwise use all
    # n = min(num_flights, len(df))
    # df = df.head(n)
    
    # Sample flights
    n = min(num_flights, len(df))
    df = df.sample(n=n, random_state=cfg.RND_SEED).sort_values('ARR_TIME').reset_index(drop=True)

    # Normalize time (t=0)
    t0 = df['ARR_TIME'].min()
    df['t_start_min'] = (df['ARR_TIME'] - t0).dt.total_seconds() / 60.0
    t_last_arrival = df['t_start_min'].max()

    print(f"[INFO] Loaded {len(df)} flights.")
    print(f"[INFO] Time window (min): 0.0 to {t_last_arrival:.1f}")
    return df, t_last_arrival


def initialize_simulation(df_flights: pd.DataFrame):
    """Initialize simulation environment and resources - synchronized with main_final.py"""
    env = simpy.Environment()
    kpi = KPIs(cfg.FLEET_SIZE)
    kpi.t_start = env.now

    # === ADDED: Flight lateness tracking fields (aligned with main_final.py) ===
    # flight_id -> t_arrival (env.now at arrival to airport system)
    kpi.flight_arrival_time = {}
    # flight_id -> t_gate_end (env.now when all ground services done at gate)
    kpi.flight_gate_end_time = {}
    # flight_id -> planned gate duration [min] from CSV (CRS_GATE_DURATION_min)
    kpi.flight_planned_gate_duration = {}
    # NOTE: AMR lateness (per job) is NOT tracked here.
    # ===========================================================================

    # Create resources
    gates = {g: simpy.Resource(env, capacity=1) for g in GATE_LABELS}
    chargers = {
        'charging_1': ChargerBank(env, 'charging_1', kpi, cfg.CHARGER_CAPACITY),
        'charging_2': ChargerBank(env, 'charging_2', kpi, cfg.CHARGER_CAPACITY),
    }

    # Create fleets
    fleets = {
        kind: AMRFleet(env, kind, size, chargers, kpi)
        for kind, size in cfg.FLEET_SIZE.items()
    }

    # Schedule flights with CRS_GATE_DURATION_min tracking
    for idx, row in df_flights.iterrows():
        flight_id = f"FL{idx:03d}"
        gate_label = row['GATE']
        start_min = float(row['t_start_min'])

        # Planned gate duration for lateness calculation
        if 'CRS_GATE_DURATION_min' not in row:
            raise KeyError("CRS_GATE_DURATION_min column is missing in arrival CSV.")
        planned_gate_min = float(row['CRS_GATE_DURATION_min'])
        kpi.flight_planned_gate_duration[flight_id] = planned_gate_min

        env.process(flight_starter(env, start_min, flight_id, gate_label, gates, fleets, kpi))

    return env, fleets, gates, chargers, kpi, df_flights


def save_flight_lateness_logs(kpi: KPIs, df_flights: pd.DataFrame, output_dir: str):
    """
    Save ONLY flight lateness logs (no AMR job lateness).
    This reads the arrival & gate end times from KPI (set inside sim_model_RL.flight_process)
    and writes log_flight_late.csv for analysis.
    """
    # Flight lateness log (log_flight_late.csv)
    flight_rows = []
    for idx, row in df_flights.iterrows():
        flight_id = f"FL{idx:03d}"

        t_arrival = kpi.flight_arrival_time.get(flight_id, None)
        t_gate_end = kpi.flight_gate_end_time.get(flight_id, None)

        if t_arrival is None or t_gate_end is None:
            continue

        # turnaround_full: arrival -> gate_end (same definition as in sim_model_RL/main_final.py)
        turnaround = t_gate_end - t_arrival
        planned_gate_min = float(row['CRS_GATE_DURATION_min'])
        late_min = max(0.0, turnaround - planned_gate_min)
        late_flag = 1 if late_min > 1e-9 else 0
        arr_sim = float(row['t_start_min'])

        flight_rows.append({
            'FLIGHT_ID': flight_id,
            'ARR_TIME_SIM': arr_sim,
            'T_ARRIVAL_ENV': t_arrival,
            'T_GATE_END_ENV': t_gate_end,
            'TURNAROUND_MIN': turnaround,
            'CRS_GATE_DURATION_min': planned_gate_min,
            'LATE': late_flag,
            'LATE_MIN': late_min,
        })

    df_flate = pd.DataFrame(flight_rows)
    df_flate_path = os.path.join(output_dir, 'log_flight_late.csv')
    df_flate.to_csv(df_flate_path, index=False)

    # Total flight lateness (sum of late minutes)
    total_flight_late_min = df_flate['LATE_MIN'].sum() if not df_flate.empty else 0.0

    return total_flight_late_min


def run_episode(episode_num: int, agent: rl_agent.ChargingAgent, df_flights: pd.DataFrame,
                t_last_arrival: float, train_mode: bool = True, return_artifacts: bool = False):
    """
    Run one episode of simulation with RL agent - tracking ONLY flight lateness

    Args:
        episode_num: Episode number
        agent: RL agent
        df_flights: Flight schedule
        t_last_arrival: Last flight arrival time
        train_mode: If True, agent trains; if False, just evaluates
        return_artifacts: If True, returns additional simulation objects (kpi, fleets, duration)

    Returns:
        If return_artifacts is False:
            (episode_reward, num_decisions, flight_lateness)
        If return_artifacts is True:
            (episode_reward, num_decisions, flight_lateness, kpi, fleets, sim_duration)
    """
    # Initialize simulation with synchronized KPI tracking
    env, fleets, gates, chargers, kpi, df_flights_copy = initialize_simulation(df_flights)
    sim_duration = t_last_arrival + cfg.SIM_BUFFER_MIN

    print(f"\n{'='*60}")
    print(f"Episode {episode_num} - Epsilon: {agent.epsilon:.3f}")
    print(f"{'='*60}")

    buffer_before = len(agent.replay_buffer)
    transitions_added = 0
    mid_train_calls = 0
    train_step_counter = 0
    episode_reward = 0.0

    # Track last decision per unit to avoid cross-AMR reward mixing
    last_decision_by_unit: dict[str, dict] = {}

    while env.now < sim_duration:
        try:
            next_evt = env.peek()
        except Exception:
            # No events scheduled; advance to sim end
            next_evt = sim_duration

        # Safety: if next_evt is at or behind current time, step once to avoid stalling
        if next_evt <= env.now:
            env.step()
            continue

        env.run(until=min(next_evt, sim_duration))

        # Process decision queues for RL charging decisions
        for fleet in fleets.values():
            while fleet.decision_queue:
                decision_info = fleet.decision_queue.pop(0)
                if decision_info.get('type') != 'charging':
                    continue

                snapshot = decision_info['snapshot']
                state_vec = decision_info['state_vector']
                decision_time = decision_info.get('decision_time', env.now)
                unit_id = decision_info['unit_id']
                action_event = decision_info['action_event']
                
                # Get flight info for reward calculation
                flight_id = decision_info.get('flight_id')
                due_time = None
                if flight_id:
                    t_arrival = kpi.flight_arrival_time.get(flight_id)
                    planned_duration = kpi.flight_planned_gate_duration.get(flight_id)
                    if t_arrival is not None and planned_duration is not None:
                        due_time = t_arrival + planned_duration

                # Get action from RL agent (epsilon-greedy etc.)
                # Now passing the correct state_vector directly to the agent
                action = agent.select_action(state_vec)

                # Resume simulation with action
                action_event.succeed(action)

                # Store transition for previous decision of the same unit if exists
                prev_decision = last_decision_by_unit.get(unit_id)
                if prev_decision is not None:
                    # Calculate reward based on the job just completed (flight_id)
                    # The reward is for the action taken at prev_decision['t']
                    reward = calculate_reward(
                        prev_decision['state_vector'],
                        prev_decision['action'],
                        decision_time,
                        due_time=due_time
                    )
                    episode_reward += float(reward)
                    if train_mode:
                        agent.store_transition(
                            prev_decision['state_vector'],
                            prev_decision['action'],
                            reward,
                            state_vec,
                            False
                        )
                        transitions_added += 1
                        train_step_counter += 1
                        if train_step_counter % 10 == 0:
                            agent.train()
                            mid_train_calls += 1

                last_decision_by_unit[unit_id] = {
                    'state_vector': state_vec,
                    'action': action,
                    't': decision_time
                }

    # Final transitions for remaining decisions
    if train_mode:
        for unit_id, prev_decision in last_decision_by_unit.items():
            # For final transition, we don't have a next job completion.
            # We can assume no tardiness penalty for the final state, or use sim_duration.
            # Here we assume 0 tardiness as the episode ended.
            final_reward = calculate_reward(
                prev_decision['state_vector'],
                prev_decision['action'],
                sim_duration,
                due_time=sim_duration # Assume met deadline at end of sim
            )
            episode_reward += float(final_reward)
            agent.store_transition(
                prev_decision['state_vector'],
                prev_decision['action'],
                final_reward,
                np.zeros_like(prev_decision['state_vector']),
                True
            )
            transitions_added += 1
            train_step_counter += 1
            if train_step_counter % 10 == 0:
                agent.train()
                mid_train_calls += 1

        # Post-episode training
        train_iters = 10
        for _ in range(train_iters):
            agent.train()
        agent.update_target_network()

    # === FLIGHT LATENESS KPI OUTPUT (no effect on reward) ===
    episode_output_dir = os.path.join(cfg.OUTPUT_DIR, f"episode_{episode_num:03d}")
    os.makedirs(episode_output_dir, exist_ok=True)

    flight_lateness = save_flight_lateness_logs(kpi, df_flights_copy, episode_output_dir)

    # Adjust episode reward with flight lateness penalty to align with performance
    flight_penalty = -flight_lateness / 1000.0
    episode_reward += flight_penalty

    print(f"\n[Episode {episode_num}] Simulation complete at t={env.now:.1f}")
    kpi.report_summary(fleets, sim_duration)

    print(f"\nFlight Lateness (sum over all flights): {flight_lateness:.2f} min")
    print(f"Adjusted Episode Reward (with flight penalty): {episode_reward:.3f}")
    # ==========================================================================

    buffer_after = len(agent.replay_buffer)
    decisions_made = max(0, buffer_after - buffer_before)

    print(f"\n[Episode {episode_num}] RL Stats:")
    print(f"  Decisions buffered: {decisions_made}")
    print(f"  Buffer Size: {len(agent.replay_buffer)}")
    print(f"  Epsilon: {agent.epsilon:.3f}")
    if train_mode:
        print(f"  Transitions added: {transitions_added}")
        print(f"  Mid-episode train calls: {mid_train_calls}")

    if return_artifacts:
        return episode_reward, decisions_made, flight_lateness, kpi, fleets, sim_duration

    return episode_reward, decisions_made, flight_lateness


def main():
    """Main training loop - tracking ONLY flight lateness (reward unchanged)"""
    print(f"\n{'='*60}")
    print("RL TRAINING - AIRPORT eGSE SIMULATION")
    print(f"Dispatching Rule: {cfg.DISPATCHING_RULE}")
    print(f"Fleet Size (Total): {cfg.TOTAL_AMR_FLEET_SIZE}")
    print(f"USE_RL_CHARGING: {cfg.USE_RL_CHARGING}")
    print(f"{'='*60}\n")

    # Enable RL charging
    cfg.USE_RL_CHARGING = True

    # Update output directory to include RL tag
    import os
    run_name = cfg.DISPATCHING_RULE
    if cfg.USE_RL_CHARGING:
        run_name += "_RL"
    cfg.OUTPUT_DIR = os.path.join("Results_TH", f"{run_name}_{cfg.TS}")

    # Setup output directory
    _setup_output_dir()

    # Load flight data
    df_flights, t_last_arrival = parse_arrivals(cfg.ARRIVAL_CSV, cfg.NUM_FLIGHTS)

    # Ensure required column exists
    if 'CRS_GATE_DURATION_min' not in df_flights.columns:
        raise KeyError("CRS_GATE_DURATION_min column is required in arrival CSV for lateness calculation")

    # Create RL agent
    agent = rl_agent.get_charging_agent()

    # Training parameters
    num_episodes = 1000
    eval_interval = 1000
    target_update_interval = 5

    # Tracking metrics (flight lateness is KPI; reward is RL metric)
    episode_rewards = []
    episode_flight_lateness = []
    best_flight_lateness = float('inf')
    best_reward = float('-inf')

    # Initialize CSV logging
    metrics_csv_path = f"{cfg.OUTPUT_DIR}/training_metrics.csv"
    with open(metrics_csv_path, 'w') as f:
        f.write("episode,reward,flight_lateness_min,epsilon\n")

    # Training loop
    for episode in range(1, num_episodes + 1):
        # Run training episode
        episode_reward, num_decisions, flight_lateness = run_episode(
            episode, agent, df_flights, t_last_arrival, train_mode=True
        )

        episode_rewards.append(episode_reward)
        episode_flight_lateness.append(flight_lateness)

        # Log to CSV immediately
        with open(metrics_csv_path, 'a') as f:
             f.write(f"{episode},{episode_reward:.4f},{flight_lateness:.4f},{agent.epsilon:.4f}\n")

        # Track best performance (for analysis; model saving commented out)
        if flight_lateness < best_flight_lateness:
            best_flight_lateness = flight_lateness
            # Save best lateness model
            agent.save_model(f"{cfg.OUTPUT_DIR}/best_model_lateness.pth")

        if episode_reward > best_reward:
            best_reward = episode_reward
            # Save best reward model
            agent.save_model(f"{cfg.OUTPUT_DIR}/best_model_reward.pth")

        # Report metrics
        avg_reward_so_far = float(np.mean(episode_rewards))
        avg_reward_last10 = float(np.mean(episode_rewards[-10:])) if len(episode_rewards) >= 10 else avg_reward_so_far
        avg_flight_lateness_last10 = (
            float(np.mean(episode_flight_lateness[-10:]))
            if len(episode_flight_lateness) >= 10
            else flight_lateness
        )

        print(f"\n[Episode {episode}] Summary:")
        print(f"  Avg Reward - Overall: {avg_reward_so_far:.3f}, Last 10: {avg_reward_last10:.3f}")
        print(f"  Avg Flight Lateness - Last 10: {avg_flight_lateness_last10:.2f} min")
        print(f"  Best Flight Lateness So Far: {best_flight_lateness:.2f} min")

        # Update target network periodically
        if episode % target_update_interval == 0:
            agent.update_target_network()
            print(f"[Episode {episode}] Target network updated")
            
        # Decay epsilon once per episode
        agent.update_epsilon()

        # Evaluation episode (no exploration)
        if episode % eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"EVALUATION at Episode {episode}")
            print(f"{'='*60}")

            old_epsilon = agent.epsilon
            agent.epsilon = 0.0  # Pure exploitation

            eval_reward, eval_decisions, eval_flight_lateness = run_episode(
                episode, agent, df_flights, t_last_arrival, train_mode=False
            )

            agent.epsilon = old_epsilon

            print(f"\nEvaluation Results:")
            print(f"  Reward: {eval_reward:.2f}")
            print(f"  Flight Lateness: {eval_flight_lateness:.2f} min")
            print(f"  Avg Training Reward (last 10): {avg_reward_last10:.2f}")
            print(f"  Avg Training Flight Lateness (last 10): {avg_flight_lateness_last10:.2f} min")

    # Final summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Final Epsilon: {agent.epsilon:.3f}")
    print(f"Final Avg Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Final Avg Flight Lateness (last 10): {np.mean(episode_flight_lateness[-10:]):.2f} min")
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Best Flight Lateness: {best_flight_lateness:.2f} min")
    print(f"Best Reward Episode: {episode_rewards.index(max(episode_rewards))+1}")
    print(f"Best Flight Lateness Episode: {episode_flight_lateness.index(min(episode_flight_lateness))+1}")

    # Save final metrics
    metrics_df = pd.DataFrame({
        'episode': range(1, num_episodes + 1),
        'reward': episode_rewards,
        'flight_lateness_min': episode_flight_lateness
    })
    # Since OUTPUT_DIR is already timestamped, we can use simple filenames
    metrics_csv_path = f"{cfg.OUTPUT_DIR}/training_reward.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nTraining metrics saved to {metrics_csv_path}")

    # Plot episode reward and last-10 moving average
    try:
        episodes_axis = list(range(1, num_episodes + 1))

        # 1) Per-episode reward
        plt.figure(figsize=(8, 4))
        plt.plot(episodes_axis, episode_rewards, label="Reward per episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Training Reward per Episode ({cfg.TS})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_path = f"{cfg.OUTPUT_DIR}/training_reward.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Reward plot saved to {plot_path}")

        # 2) Last-10 moving average reward
        window = 10
        moving_avg = []
        for i in range(len(episode_rewards)):
            start = max(0, i - window + 1)
            moving_avg.append(np.mean(episode_rewards[start:i + 1]))

        plt.figure(figsize=(8, 4))
        plt.plot(episodes_axis, moving_avg, label="Last-10 avg reward", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Reward (avg of last 10)")
        plt.title(f"Last-10 Average Training Reward ({cfg.TS})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_path_ma = f"{cfg.OUTPUT_DIR}/training_reward_last10.png"
        plt.savefig(plot_path_ma, dpi=150)
        plt.close()
        print(f"Last-10 reward plot saved to {plot_path_ma}")

        # 3) Delay moving-average (last-10) plot with timestamp in title
        moving_delay = []
        for i in range(len(episode_flight_lateness)):
            start = max(0, i - window + 1)
            moving_delay.append(np.mean(episode_flight_lateness[start:i + 1]))

        plt.figure(figsize=(8, 4))
        plt.plot(episodes_axis, moving_delay, label="Flight lateness (last 10 avg)")
        plt.xlabel("Episode")
        plt.ylabel("Total flight delay (min)")
        plt.title(f"Flight Delay (Last-10 Avg) ({cfg.TS})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_path_delay = f"{cfg.OUTPUT_DIR}/training_delay_last10.png"
        plt.savefig(plot_path_delay, dpi=150)
        plt.close()
        print(f"Delay plot saved to {plot_path_delay}")
    except Exception as e:
        print(f"[WARN] Failed to save reward plots: {e}")

    # Save final model
    agent.save_model(f"{cfg.OUTPUT_DIR}/final_model.pth")
    _plot_gate_gantt(cfg.TS)
    print(f"Results saved to {cfg.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
