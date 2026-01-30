# ==============================================================================
# ===== AMR 클래스와 SimPy 프로세스 =====
# ==============================================================================

import os
import sys
import simpy
import random
import statistics as stats
from datetime import datetime
from typing import Optional

# config와 routing에서 필요한 모든 변수와 함수를 가져옵니다.
import config as cfg
from routing import (
    _get_path_waypoints, _calculate_path_distance_and_time,
    DEPOT_LABEL, CHARGER_LABELS, NODE_POS
)
# reporting에서 LOG와 update_state 함수를 가져옵니다.
from reporting import LOG, update_state
# RL agent 모듈
import rl_agent
import numpy as np

# biding_1 모듈 (Bidding Rule)
try:
    import bidding_1
except ImportError:
    bidding_1 = None

# Diagnostics: timestamped log file for shortage/override tracing
DIAG_LOG_PATH = os.path.join(
    cfg.OUTPUT_DIR,
    f"diag_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)


def diag_log(message: str):
    """Append lightweight diagnostics with wallclock timestamp."""
    try:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(DIAG_LOG_PATH, "a") as f:
            f.write(f"[{ts}] {message}\n")
    except Exception as e:
        print(f"[WARN] diag_log failed: {e}")


class AMRUnit:
    """Represents a single AMR unit."""
    def __init__(self, unit_id: int, kind: str, capacity_kwh: float):
        self.unit_id = unit_id
        self.kind = kind
        self.global_id = f"{kind}_{unit_id}"

        self.soc_kwh = capacity_kwh
        self.capacity_kwh = capacity_kwh
        self.location = DEPOT_LABEL

        self.total_work_time = 0.0
        self.num_tasks = 0

        self.time_tracker = {"t_last_update": 0.0, "state": "IDLE"}

    @property
    def soc_percent(self) -> float:
        return self.soc_kwh / self.capacity_kwh

    def _update_time_kpi(self, env: simpy.Environment, new_state: str, kpi):
        """Updates the time-tracking KPI log."""
        t_now = env.now
        duration = t_now - self.time_tracker["t_last_update"]
        if duration > 0:
            last_state = self.time_tracker["state"]
            kpi.amr_time_log[(self.kind, self.unit_id)][last_state] += duration

        self.time_tracker["state"] = new_state
        self.time_tracker["t_last_update"] = t_now

    def consume_energy(self, duration_min: float, power_kw: float, kpi):
        """Consumes energy and updates SOC."""
        used_kwh = power_kw * (duration_min / 60.0)
        self.soc_kwh = max(0.0, self.soc_kwh - used_kwh)
        kpi.total_energy_consumed += used_kwh

class ChargerBank:
    """Represents a charging station with configurable capacity."""
    def __init__(self, env: simpy.Environment, name: str, kpi, capacity: int):
        self.env = env
        self.name = name
        self.res = simpy.Resource(env, capacity=capacity)
        self.kpi = kpi
        self.time_tracker = {"t_last_update": 0.0, "queue": 0}

    def _update_util_kpi(self):
        """Updates charger utilization KPI."""
        t_now = self.env.now
        queue = len(self.res.queue)

        if self.time_tracker["queue"] > 0:
            duration = t_now - self.time_tracker["t_last_update"]
            self.kpi.charger_time_log[self.name]["BUSY"] += duration

        self.time_tracker["t_last_update"] = t_now
        self.time_tracker["queue"] = queue

class AMRFleet:
    """Manages a fleet of AMRUnits of a specific kind (e.g., 'GPU')."""

    def __init__(self, env: simpy.Environment, kind: str, size: int,
                 chargers: dict, kpi, rl_mode=False):
        """rl_mode가 True면 RL 환경에서 사용"""
        self.rl_mode = rl_mode
        self.gym_env = None  # RL 모드일 때 설정
        self._init_fleet(env, kind, size, chargers, kpi)

    def _init_fleet(self, env: simpy.Environment, kind: str, size: int,
                    chargers: dict, kpi):
        self.env = env
        self.kind = kind

        # 동현 수정
        # 각 AMR마다 모두 다른 용량의 battery capacity 적용
        capacity = cfg.BATTERY_CAP_KWH.get(kind, cfg.DEFAULT_BATTERY_CAP_KWH)

        self.units = [AMRUnit(i, kind, capacity_kwh=capacity) for i in range(size)]

        # Use a list to maintain deterministic order (FIFO based on insertion)
        self.available = [u.global_id for u in self.units]
        self.unit_map = {u.global_id: u for u in self.units}

        self.lock = simpy.Resource(env, capacity=1)
        self.chargers = chargers
        self.kpi = kpi

        # Throttle diagnostic spam for eligibility shortages
        self.last_insufficient_log_time = -1.0

        # RL decision queue (main-driven)
        self.decision_queue = []  # each item: {'type', 'snapshot', 'state_vector', 'unit_id', 'unit', 'action_event'}
    
    def set_gym_env(self, gym_env):
        """RL 모드에서 Gym 환경 연결"""
        self.gym_env = gym_env

    def _travel(self, unit: AMRUnit, dst: str, flight_id: str = None):
        """Generator for AMR travel using (x, y) router."""
        if unit.location == dst:
            yield self.env.timeout(0)
            return

        unit._update_time_kpi(self.env, "TRAVEL", self.kpi)
        t_start = self.env.now
        src = unit.location
        
        try:
            waypoints = _get_path_waypoints(src, dst)
        except NotImplementedError:
            print(f"[ERROR] No path defined: {src} -> {dst}. Halting.")
            yield self.env.timeout(99999)
            return
            
        distance, travel_time = _calculate_path_distance_and_time(waypoints)
        
        if travel_time > 0:
            unit.consume_energy(travel_time, cfg.TRAVEL_CONSUME_POWER_KW, self.kpi)
            yield self.env.timeout(travel_time)
        
        unit.location = dst
        unit.total_work_time += travel_time
        self.kpi.total_travel_distance += distance

        LOG.log_amr(
            self.env.now, unit.global_id, self.kind, "travel_end",
            t_start=t_start, src=src, dst=dst, flight_id=flight_id, 
            path=waypoints
        )

    def _service(self, unit: AMRUnit, duration_min: float, flight_id: str, task: str):
        """Generator for AMR service, consuming energy."""
        unit._update_time_kpi(self.env, "SERVICE", self.kpi)
        t_start = self.env.now

        # 동현 수정
        # 각 AMR 마다 각기 다른 서비스 power 사용
        power_kw = cfg.SERVICE_CONSUME_POWER_KW.get(unit.kind, cfg.DEFAULT_SERVICE_CONSUME_POWER_KW)
        unit.consume_energy(duration_min, power_kw, self.kpi)

        yield self.env.timeout(duration_min)

        unit.total_work_time += duration_min
        unit.num_tasks += 1

        LOG.log_amr(
            self.env.now, unit.global_id, self.kind, "service_end",
            t_start=t_start, duration=duration_min, flight_id=flight_id, task=task
        )

    def _charge(self, unit: AMRUnit, charger_name: str):
        """Generator for charging at a specific charger."""
        charger = self.chargers[charger_name]
        
        yield self.env.process(self._travel(unit, charger_name))

        unit._update_time_kpi(self.env, "Q_CHARGE", self.kpi)
        t_queue_start = self.env.now
        charger._update_util_kpi()
        
        with charger.res.request() as creq:
            yield creq
            
            charger._update_util_kpi()
            t_charge_start = self.env.now
            unit._update_time_kpi(self.env, "CHARGING", self.kpi)
            LOG.log_amr(
                t_charge_start, unit.global_id, self.kind, "charge_start",
                charger=charger_name, queue_time=(t_charge_start - t_queue_start)
            )

            need_kwh = max(0.0, unit.capacity_kwh - unit.soc_kwh)
            if need_kwh > 0:
                hours_to_charge = need_kwh / cfg.CHARGE_POWER_KW
                duration_min = hours_to_charge * 60.0
                yield self.env.timeout(duration_min)
                
                unit.soc_kwh = unit.capacity_kwh
                self.kpi.total_charge_events += 1
                self.kpi.total_charge_kwh += need_kwh

        charger._update_util_kpi()
        LOG.log_amr(
            self.env.now, unit.global_id, self.kind, "charge_end",
            charger=charger_name
        )

    def _find_shortestQ_charger(self, unit: AMRUnit) -> tuple[str, float]:
        """Finds the BEST charger (shortest queue, then shortest time)."""
        options = []
        
        for name, charger_bank in self.chargers.items():
            try:
                queue_len = len(charger_bank.res.queue)
                waypoints = _get_path_waypoints(unit.location, name)
                _, time = _calculate_path_distance_and_time(waypoints)
                options.append((queue_len, time, name))
            except NotImplementedError:
                continue
        
        if not options:
            return None, float('inf')
            
        options.sort(key=lambda x: (x[0], x[1]))
        
        best_queue, best_time, best_name = options[0]
        return best_name, best_time
        
    def _get_eligible_units(self, required_kwh: float) -> list:
        """Finds units at DEPOT with enough charge."""
        eligible_units = []
        # Iterate over self.available which is now a list maintaining FIFO order
        for unit_id in self.available:
            unit = self.unit_map[unit_id]
            if unit.location == DEPOT_LABEL and unit.soc_kwh >= required_kwh:
                eligible_units.append(unit)
        return eligible_units

    def _select_units_by_rule(self, eligible: list, n: int, task: str = "", gate_label: Optional[str] = None) -> list:
        """Applies the configured dispatching rule."""
        if len(eligible) < n:
            return []

        if cfg.DISPATCHING_RULE == 'FIFO':
            selected = eligible[:n]
        
        elif cfg.DISPATCHING_RULE == 'RANDOM':
            selected = random.sample(eligible, n)
            
        elif cfg.DISPATCHING_RULE == 'LEAST_UTILIZED':
            eligible.sort(key=lambda u: (u.total_work_time, u.num_tasks))
            selected = eligible[:n]

        elif cfg.DISPATCHING_RULE == 'BIDDING':
            # Use routing-based bid: lower score is better.
            # If gate_label is missing, fall back to FIFO order.
            if gate_label is None:
                selected = eligible[:n]
            else:
                ranked = bidding_1.rank_units_by_bid(eligible, gate_label)
                selected = [u for u, _ in ranked[:n]]
                

        else:
            raise ValueError(f"Unknown DISPATCHING_RULE: {cfg.DISPATCHING_RULE}")

        return selected

    def request_units(self, n: int, required_kwh: float, task: str, gate_label: Optional[str] = None):
        """Generator to request 'n' available units based on rules or RL.

        gate_label is used only for bidding-based dispatch to compute travel-based scores.
        """
        if self.rl_mode and self.gym_env is not None:
            # RL 모드: Agent의 선택을 기다림
            selected = yield self.env.process(self._request_units_rl(n, required_kwh, task, gate_label))
        else:
            # 일반 모드: 규칙 기반 선택
            selected = yield self.env.process(self._request_units_rule(n, required_kwh, task, gate_label))
        return selected
    
    def _request_units_rl(self, n: int, required_kwh: float, task: str, gate_label: Optional[str] = None):
        """RL 모드: Agent에게 decision 요청"""
        while True:
            with self.lock.request() as req:
                yield req
                
                eligible = self._get_eligible_units(required_kwh)
                
                if len(eligible) < n:
                    now = self.env.now
                    if self.last_insufficient_log_time < 0 or now - self.last_insufficient_log_time >= 1.0:
                        depot_units = [
                            self.unit_map[uid]
                            for uid in self.available
                            if self.unit_map[uid].location == DEPOT_LABEL
                        ]
                        socs = sorted((u.soc_percent for u in depot_units), reverse=True)
                        diag_log(
                            f"[{self.kind}] insufficient eligible units for {task} at t={now:.2f}"
                            f" required_kwh={required_kwh:.2f} available_depot={len(depot_units)}"
                            f" eligible={len(eligible)} soc_top5={[round(s, 3) for s in socs[:5]]}"
                        )
                        self.last_insufficient_log_time = now
                    
                    # 강제 충전: 부족한 유닛 수만큼 디포트 유닛 중 SoC 낮은 순으로 충전 시작
                    shortage = n - len(eligible)
                    depot_units = [
                        self.unit_map[uid]
                        for uid in self.available
                        if self.unit_map[uid].location == DEPOT_LABEL
                    ]
                    low_soc_units = sorted(depot_units, key=lambda u: u.soc_percent)[:shortage]
                    for unit in low_soc_units:
                        charger_name, _ = self._find_shortestQ_charger(unit)
                        if charger_name:
                            diag_log(f"Force charging {unit.global_id} ({self.kind}) at t={now:.2f}, soc={unit.soc_percent:.3f}")
                            self.env.process(self._charge(unit, charger_name))
                    
                    yield self.env.timeout(0.5)
                    continue
                
                # RL Agent에게 결정 요청
                decision_info = {
                    'fleet_kind': self.kind,
                    'task': task,
                    'candidates': eligible,
                    'required_kwh': required_kwh,
                    'n_required': n
                }
                
                # Gym 환경에 알림
                self.gym_env.post_decision_request(decision_info)
                
                # Agent의 응답 대기 (interrupt로 전달됨)
                try:
                    # 무한 대기
                    yield self.env.timeout(float('inf'))
                except simpy.Interrupt as interrupt:
                    # interrupt.cause에 선택된 인덱스들이 담겨있음
                    selected_indices = interrupt.cause
                    
                    if len(selected_indices) != n:
                        print(f"[WARN] Expected {n} units, got {len(selected_indices)}")
                        selected_indices = selected_indices[:n]
                    
                    selected_units = [eligible[i] for i in selected_indices]
                    
                    for unit in selected_units:
                        self.available.remove(unit.global_id)
                        unit._update_time_kpi(self.env, "Q_TASK", self.kpi)
                    
                    LOG.log_amr(
                        self.env.now, f"Fleet_{self.kind}", self.kind, "dispatch_success_rl",
                        task=task, n=n
                    )
                    return selected_units
    
    def _request_units_rule(self, n: int, required_kwh: float, task: str, gate_label: Optional[str] = None):
        """일반 모드: 규칙 기반 선택"""
        def _get_units():
            while True:
                with self.lock.request() as req:
                    yield req
                    
                    eligible = self._get_eligible_units(required_kwh)
                    selected_units = self._select_units_by_rule(eligible, n, task, gate_label)
                    
                    if len(selected_units) == n:
                        for unit in selected_units:
                            self.available.remove(unit.global_id)
                            unit._update_time_kpi(self.env, "Q_TASK", self.kpi)
                        return selected_units
                
                yield self.env.timeout(0.1)

        t_request = self.env.now
        selected = yield self.env.process(_get_units())
        
        wait_time = self.env.now - t_request
        LOG.log_amr(
            self.env.now, f"Fleet_{self.kind}", self.kind, "dispatch_success",
            task=task, wait_time=wait_time, n=n
        )
        return selected

    def _create_state_vector(self, snapshot: dict, unit_id: str) -> np.ndarray:
        """
        Create 6-dim state vector for the specific target unit.
        Matches Formula: s = [SoC, Q_Ch1, Q_Ch2, Dist_Ch1, Dist_Ch2, Work_Time]
        """
        # 1. Find Target Unit
        target_amr = None
        amr_list = snapshot.get('amr_states', [])
        
        for amr in amr_list:
            if amr['global_id'] == unit_id:
                target_amr = amr
                break
        
        if target_amr is None:
             return np.zeros(6, dtype=np.float32)

        # 2. Extract Data
        # [1] SoC
        my_soc = target_amr.get('soc_percent', 0.0)
        
        # [6] Work Time (Phi)
        my_work_time_norm = target_amr.get('total_work_time', 0.0) / 1000.0
        
        # Location for Distance Calc
        my_xy = target_amr.get('location_xy', (0, 0))
        pos_ch1 = NODE_POS.get('charging_1', (0,0))
        pos_ch2 = NODE_POS.get('charging_2', (0,0))
        
        # [4, 5] Distance
        dist_ch1 = ((my_xy[0]-pos_ch1[0])**2 + (my_xy[1]-pos_ch1[1])**2)**0.5
        dist_ch2 = ((my_xy[0]-pos_ch2[0])**2 + (my_xy[1]-pos_ch2[1])**2)**0.5
        
        # Scale Distance (approx max 20)
        dist_ch1 /= 20.0
        dist_ch2 /= 20.0

        # [2, 3] Charger Queues
        charger_states = snapshot.get('charger_states', {})
        q_ch1 = float(charger_states.get('charging_1', 0)) / 5.0
        q_ch2 = float(charger_states.get('charging_2', 0)) / 5.0

        # Order: [SoC, Q1, Q2, D1, D2, Phi]
        state_vec = [
            my_soc, q_ch1, q_ch2, dist_ch1, dist_ch2, my_work_time_norm
        ]
        return np.array(state_vec, dtype=np.float32)

    def release_units(self, units: list, task_end_location: str, all_fleets: dict, flight_id: str = None):
        """Generator to release units, triggering charge/return logic."""

        def _unit_return_logic(unit: AMRUnit, all_fleets: dict):
            # Get state snapshot (한 번만!)
            snapshot = update_state(self.env, "amr_task_end", self.kpi, all_fleets)
            unit.location = task_end_location
            
            # RL 기반 충전 결정 (또는 기존 규칙)
            if hasattr(cfg, 'USE_RL_CHARGING') and cfg.USE_RL_CHARGING:
                # 상태 벡터 직접 생성 (RL agent 의존성 제거)
                state_vec = self._create_state_vector(snapshot, unit.global_id)

                # Hard safety override: force charge when SoC is critically low
                if unit.soc_percent <= 0.10:
                    charger_name, _ = self._find_shortestQ_charger(unit)
                    if charger_name:
                        yield self.env.process(self._charge(unit, charger_name))
                        # After forced charge, return to depot if needed
                        if unit.location != DEPOT_LABEL:
                            yield self.env.process(self._travel(unit, DEPOT_LABEL))
                        unit._update_time_kpi(self.env, "IDLE", self.kpi)
                        with self.lock.request() as req:
                            yield req
                            self.available.append(unit.global_id)
                        return
                    else:
                        print(f"[WARN] {unit.global_id} below 10% but no charger path found. Proceeding with RL decision.")

                # 메인으로 결정 요청 (action_event로 대기)
                action_event = self.env.event()
                self.decision_queue.append({
                    'type': 'charging',
                    'snapshot': snapshot,
                    'state_vector': state_vec,
                    'unit_id': unit.global_id,
                    'unit': unit,
                    'decision_time': self.env.now,
                    'flight_id': flight_id,  # Pass flight_id for reward calculation
                    'action_event': action_event
                })

                # 메인에서 action_event.succeed(action) 호출할 때까지 대기
                action = yield action_event

                # [DEBUG] Print decisions occasionally to debug Ch1/Ch2 balance
                if random.random() < 0.01: # Print 1% of decisions
                    print(f"[DEBUG] {unit.global_id} Action: {action} (SoC: {unit.soc_percent:.2f})")

                # action 실행
                if action == 1:
                    yield self.env.process(self._charge(unit, list(self.chargers.keys())[0]))
                elif action == 2:
                    yield self.env.process(self._charge(unit, list(self.chargers.keys())[1]))
            else:
                # 기존 규칙 기반
                if unit.soc_percent < cfg.CHARGE_TRIGGER_SOC:
                    charger_name, _ = self._find_shortestQ_charger(unit)
                    if charger_name:
                        yield self.env.process(self._charge(unit, charger_name))
                    else:
                        print(f"[WARN] {unit.global_id} needs charge but no path found.")
            
            if unit.location != DEPOT_LABEL:
                yield self.env.process(self._travel(unit, DEPOT_LABEL))
            
            unit._update_time_kpi(self.env, "IDLE", self.kpi)
            with self.lock.request() as req:
                yield req
                # Append to list to maintain FIFO order of availability
                if unit.global_id not in self.available:
                    self.available.append(unit.global_id)

        for u in units:
            self.env.process(_unit_return_logic(u, all_fleets))
        yield self.env.timeout(0)

# ==============================================================================
# ===== SIMULATION PROCESS LOGIC =====
# ==============================================================================

def _task_process(
    env: simpy.Environment, 
    flight_id: str, 
    gate_label: str,
    task_name: str, 
    fleets: dict,
    kpi,
    gpu_arrived_event: simpy.Event,
    gpu_assigned_event: simpy.Event = None,
    baggage_out_done_event: simpy.Event = None
):
    """Core generator for a single eGSE task (e.g., 'FUEL')."""
    if gpu_assigned_event:
        yield gpu_assigned_event

    fleet_name = cfg.TASK_TO_FLEET_MAP[task_name]
    fleet = fleets[fleet_name]
    num_units = cfg.REQUIRED_UNITS[task_name]
    service_duration = cfg.SERVICE_TIMES[task_name]

    # 1. Calculate required energy
    try:
        _, t_to_gate = _calculate_path_distance_and_time(_get_path_waypoints(DEPOT_LABEL, gate_label))
        _, t_to_depot = _calculate_path_distance_and_time(_get_path_waypoints(gate_label, DEPOT_LABEL))
    except NotImplementedError:
        print(f"[ERROR] Cannot calculate energy for {task_name} at {gate_label}.")
        return
        
    travel_time = t_to_gate + t_to_depot
    service_power_kw = cfg.SERVICE_CONSUME_POWER_KW.get(fleet_name, cfg.DEFAULT_SERVICE_CONSUME_POWER_KW)

    required_kwh = (travel_time * cfg.TRAVEL_CONSUME_POWER_KW + 
                    service_duration * service_power_kw) / 60.0
    
    # 2. Request units
    units = yield env.process(fleet.request_units(num_units, required_kwh, task_name, gate_label))
    unit = units[0]
    LOG.log_amr(
        env.now, unit.global_id, fleet_name, "dispatch_assigned", 
        flight_id=flight_id, task=task_name
    )
    
    # 3. Travel to gate
    yield env.process(fleet._travel(unit, gate_label, flight_id=flight_id))
    
    # 4. === EVENT SYNCHRONIZATION ===
    unit._update_time_kpi(env, "Q_TASK", kpi) 
    t_wait_start = env.now

    # Wait for GPU
    yield gpu_arrived_event
    wait_time = env.now - t_wait_start
    kpi.gpu_arrival_wait_times.append(wait_time)
    
    LOG.log_amr(
        env.now, unit.global_id, fleet_name, "gpu_wait_over", 
        flight_id=flight_id, wait_time=wait_time
    )

    if task_name == 'BAGGAGE_IN':
        # Wait for BAGGAGE_OUT
        t_bag_wait_start = env.now
        yield baggage_out_done_event
        wait_time_bag = env.now - t_bag_wait_start
        kpi.baggage_in_wait_times.append(wait_time_bag)
        unit._update_time_kpi(env, "Q_TASK", kpi) # Log time waiting for BagOut
        LOG.log_amr(
            env.now, unit.global_id, fleet_name, "bag_out_wait_over", 
            flight_id=flight_id, wait_time=wait_time_bag
        )

    # 5. Perform Service
    if service_duration > 0:
        yield env.process(fleet._service(unit, service_duration, flight_id, task_name))
        
    # 6. Signal completion
    if task_name == 'BAGGAGE_OUT':
        baggage_out_done_event.succeed()
        LOG.log_amr(
            env.now, unit.global_id, fleet_name, "bag_out_done_signal", 
            flight_id=flight_id
        )

    LOG.log_flight(env.now, flight_id, "task_completed", task=task_name)

    # 7. Release unit
    env.process(fleet.release_units(units, gate_label, fleets, flight_id=flight_id))

def flight_process(env: simpy.Environment, flight_id: str, gate_label: str, 
                   gates: dict, fleets: dict, kpi):
    """Main generator for a single flight."""
    
    # 1. Wait for gate
    t_arrival = env.now
    LOG.log_flight(t_arrival, flight_id, "gate_queued", gate=gate_label)

    # 동현 수정
    # flight 도착 시각 기록 -> 나중에 비행기 늦은거 계산할때 사용하려고 만듬
    kpi.flight_arrival_time[flight_id] = t_arrival
    
    with gates[gate_label].request() as greq:
        yield greq
        
        t_gate_start = env.now
        gate_wait = t_gate_start - t_arrival
        kpi.gate_wait_times.append(gate_wait)
        LOG.log_flight(t_gate_start, flight_id, "gate_start", gate=gate_label, wait_time=gate_wait)

        update_state(env, "flight_arrival", kpi, fleets)

        # 2. Create event triggers
        gpu_arrived = env.event()
        gpu_assigned = env.event()
        baggage_out_done = env.event()

        # 3. Handle GPU and OTHER tasks in parallel
        gpu_fleet = fleets['GPU']

        # 3a. Define GPU sub-process
        def _gpu_main_task_process(env, flight_id, gate_label, kpi_inst, t_gate_arrival, gpu_arrived_event, gpu_assigned_event, fleet_to_use):
            """Process for the GPU's entire lifecycle at the gate."""
            
            # 1. Request GPU Unit
            try:
                _, t_to_gate = _calculate_path_distance_and_time(_get_path_waypoints(DEPOT_LABEL, gate_label))
            except NotImplementedError:
                 t_to_gate = 1.0
            required_kwh = (t_to_gate * cfg.TRAVEL_CONSUME_POWER_KW) / 60.0
            
            units = yield env.process(fleet_to_use.request_units(1, required_kwh, 'GPU', gate_label))
            gpu_unit = units[0]

            # Signal that GPU has been assigned effectively reserving it
            if gpu_assigned_event:
                gpu_assigned_event.succeed()

            # 2. Travel
            yield env.process(fleet_to_use._travel(gpu_unit, gate_label, flight_id=flight_id))
            
            # 3. Arrive & Signal
            gpu_arrived_event.succeed()
            LOG.log_amr(env.now, gpu_unit.global_id, 'GPU', "gpu_arrived_signal", flight_id=flight_id)
            
            # 4. Return unit AND service start time
            t_service_start = env.now
            gpu_unit._update_time_kpi(env, "SERVICE", kpi_inst)
            return (gpu_unit, t_service_start)
        
        # 3b. Start all tasks in parallel
        tasks_to_start = [
            'FUEL', 'WATER', 'CLEAN', 'CATERING', 'BAGGAGE_OUT', 'BAGGAGE_IN'
        ]
        
        gpu_process = env.process(
            _gpu_main_task_process(env, flight_id, gate_label, kpi, t_gate_start, gpu_arrived, gpu_assigned, gpu_fleet)
        )
        
        all_other_task_processes = []
        for task in tasks_to_start:
            proc = env.process(
                _task_process(
                    env, flight_id, gate_label, task, fleets, kpi,
                    gpu_arrived_event=gpu_arrived,
                    gpu_assigned_event=gpu_assigned,
                    baggage_out_done_event=baggage_out_done
                )
            )
            all_other_task_processes.append(proc)
        
        # 4. Wait for ALL OTHER tasks to complete
        yield simpy.AllOf(env, all_other_task_processes)


        # 5. Get GPU unit and consume service energy
        (gpu_unit, t_service_start) = yield gpu_process
        t_service_end = env.now
        
        gpu_service_duration = t_service_end - t_service_start
        if gpu_service_duration > 0:
            # 동현 수정
            # GPU 배터리 사용에 대한 maximum (35분) 정해두기
            max_power_minutes = cfg.SERVICE_TIMES.get('GPU', gpu_service_duration)
            effective_minutes = min(gpu_service_duration, max_power_minutes)

            if effective_minutes > 0:
                power_kw = cfg.GPU_CONFIG['SERVICE_CONSUME_POWER_KW']
                gpu_unit.consume_energy(effective_minutes, power_kw, kpi)

        LOG.log_flight(env.now, flight_id, "task_completed", task='GPU')
        env.process(gpu_fleet.release_units([gpu_unit], gate_label, fleets, flight_id=flight_id))

        # 6. All services done, release gate
        t_gate_end = env.now

        # (1) 게이트에 실제로 붙어 있던 시간 (참고용)
        turnaround_gate = t_gate_end - t_gate_start
        kpi.flight_turnaround_times.append(turnaround_gate)

        # (2) 우리가 lateness 정의에 사용할 전체 구간: arrival ~ gate_end
        t_arrival = kpi.flight_arrival_time[flight_id]
        turnaround_full = t_gate_end - t_arrival

        # flight gate 작업 종료 시각 기록
        kpi.flight_gate_end_time[flight_id] = t_gate_end

        # ★ per-flight CRS_GATE_DURATION_min 기준으로 늦음 판정 (model_final.py와 동일)
        planned_gate_min = kpi.flight_planned_gate_duration.get(
            flight_id,
            cfg.TARGET_TURNAROUND_MINUTES  # fallback
        )

        if turnaround_full > planned_gate_min:
            kpi.flight_delays_count += 1
            kpi.flight_delay_minutes.append(turnaround_full - planned_gate_min)

        LOG.log_flight(t_gate_end, flight_id, "gate_end", gate=gate_label, duration=turnaround_gate)


def flight_starter(env, t_start_min: float, *args):
    """Schedules a flight_process to start at a specific time."""
    if t_start_min > env.now:
        yield env.timeout(t_start_min - env.now)
    env.process(flight_process(env, *args))
