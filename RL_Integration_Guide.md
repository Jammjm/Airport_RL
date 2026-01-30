# RL 통합 가이드 - 최소 수정 방식

## 📋 개요

기존 시뮬레이션 코드에 **최소한의 수정**으로 RL을 통합하는 방법입니다.

### 핵심 아이디어
- **별도 모듈**: `rl_agent.py`를 만들어 RL 로직 분리
- **기존 구조 활용**: `update_state()` 호출 시점을 그대로 사용
- **점진적 적용**: 규칙 기반 → RL 기반으로 단계적 전환 가능

---

## 🎯 RL Agent의 역할

### 1️⃣ 충전 결정 (Charging Decision)
**시점**: AGV 작업 완료 후 (`amr_task_end` trigger)

```python
# sim_model_RL.py의 release_units()에서
charger_name = rl_agent.decide_charging(unit, chargers, current_time)
# Returns: None (no charge) or "C1" or "C2"
```

**State**:
- Battery level
- Charger queue lengths
- AGV workload
- Distance to chargers

**Action**:
- 0: No charging
- 1: Charge at Ch1
- 2: Charge at Ch2

---

### 2️⃣ 비딩 결정 (Dispatch/Bidding Decision)
**시점**: 비행기 도착 시 (`flight_arrival` trigger)

```python
# sim_model_RL.py의 _select_units_by_rule()에서
if cfg.DISPATCHING_RULE == 'RL_BIDDING':
    selected = rl_agent.select_units_by_rl(eligible, n, task_info)
```

**특징**:
- 미래 상태를 상상 (imaginary state)
- 작업 완료 후 충전이 필요한지 RL로 예측
- 예측된 충전 시간을 비딩값에 반영

---

## 📂 파일 구조

```
simulation/
├── config.py              # 설정 (RL 옵션 추가)
├── model.py               # 원본 (수정 안함)
├── sim_model_RL.py        # RL 통합 버전 (최소 수정)
├── reporting.py           # 원본 (수정 안함)
├── routing.py             # 원본 (수정 안함)
│
├── rl_agent.py            # ★ 새로 추가: RL 로직
├── test_rl_integration.py # ★ 새로 추가: 테스트
│
├── main.py                # 원본 실행
└── main_RL.py             # RL 실행 (선택)
```

---

## 🔧 수정 내역

### 1. `config.py` (3줄 추가)
```python
# RL 옵션 추가
USE_RL_CHARGING = False  # True로 설정하면 RL 충전 결정
DISPATCHING_RULE = 'RL_BIDDING'  # RL 기반 비딩
RL_AGENT_MODE = 'rule'  # 'rule' or 'rl'
```

### 2. `sim_model_RL.py` (3곳 수정)

#### (1) Import 추가
```python
import rl_agent
```

#### (2) Dispatch 로직에 RL 옵션 추가
```python
def _select_units_by_rule(self, eligible, n, task=""):
    # ... 기존 코드 ...
    
    elif cfg.DISPATCHING_RULE == 'RL_BIDDING':
        task_info = {...}
        selected = rl_agent.select_units_by_rl(eligible, n, task_info)
```

#### (3) 충전 로직에 RL 옵션 추가
```python
def _unit_return_logic(unit, all_fleets):
    update_state(env, "amr_task_end", kpi, all_fleets)
    
    if cfg.USE_RL_CHARGING:
        charger_name = rl_agent.decide_charging(unit, chargers, env.now)
    else:
        # 기존 규칙
        if unit.soc_percent < cfg.CHARGE_TRIGGER_SOC:
            charger_name = find_shortest_charger(...)
```

### 3. `rl_agent.py` (새 파일)
- `ChargingAgent`: 충전 결정
- `DispatchAgent`: 비딩 결정
- Helper functions

---

## 🚀 사용 방법

### Step 1: 테스트 실행
```bash
cd simulation
python test_rl_integration.py --test all
```

**출력 예시**:
```
Test Case: Rule-based (Original)
  - Dispatch Rule: RANDOM
  - RL Charging: False
  ✓ Completed!
    - Flights Handled: 5
    - Delays: 0
    - Total Energy: 245.3 kWh

Test Case: RL Bidding (Dispatch)
  - Dispatch Rule: RL_BIDDING
  - RL Charging: False
  ✓ Completed!

Test Case: Full RL (Bidding + Charging)
  - Dispatch Rule: RL_BIDDING
  - RL Charging: True
  ✓ Completed!
```

### Step 2: 모드별 실행

#### 모드 A: 규칙 기반 (원본)
```python
# config.py
DISPATCHING_RULE = 'RANDOM'
USE_RL_CHARGING = False
```

#### 모드 B: RL 충전만 사용
```python
# config.py
DISPATCHING_RULE = 'RANDOM'  # 기존 규칙
USE_RL_CHARGING = True       # RL 충전
```

#### 모드 C: RL 비딩만 사용
```python
# config.py
DISPATCHING_RULE = 'RL_BIDDING'  # RL 비딩
USE_RL_CHARGING = False          # 기존 충전
```

#### 모드 D: Full RL
```python
# config.py
DISPATCHING_RULE = 'RL_BIDDING'  # RL 비딩
USE_RL_CHARGING = True           # RL 충전
```

### Step 3: 일반 시뮬레이션 실행
```bash
# sim_model_RL.py를 import하도록 수정된 main.py 실행
python main.py
```

---

## 📊 Decision Points 요약

| Decision Point | File | Function | Trigger | RL Agent |
|---|---|---|---|---|
| **AGV 작업 완료** | sim_model_RL.py | `release_units()` → `_unit_return_logic()` | `update_state("amr_task_end")` | `decide_charging()` |
| **비행기 도착** | sim_model_RL.py | `flight_process()` → `request_units()` → `_select_units_by_rule()` | `update_state("flight_arrival")` | `select_units_by_rl()` |

---

## 🔄 작동 흐름

### 충전 결정 흐름
```
1. AGV가 작업 완료
   ↓
2. release_units() 호출
   ↓
3. update_state("amr_task_end") ← State 업데이트
   ↓
4. if USE_RL_CHARGING:
       charger = rl_agent.decide_charging(...)
   ↓
5. _charge() 또는 depot으로 직행
```

### 비딩 흐름
```
1. 비행기 도착
   ↓
2. flight_process() 실행
   ↓
3. update_state("flight_arrival") ← State 업데이트
   ↓
4. request_units() 호출
   ↓
5. _select_units_by_rule() 호출
   ↓
6. if DISPATCHING_RULE == 'RL_BIDDING':
       for each AGV:
           # 미래 상태 예측
           future_state = imagine_after_task(...)
           # RL로 충전 예측
           will_charge = rl_agent.predict_charging(future_state)
           # 비딩값 계산
           bid = base_score + charging_cost(will_charge)
   ↓
7. 최저 비딩값 AGV 선택
```

---

## 🎓 RL 학습 (향후)

### 현재: Rule-based
```python
rl_agent.get_charging_agent(mode='rule')  # Heuristic
```

### 향후: DQN/PPO 학습 후
```python
rl_agent.get_charging_agent(mode='rl')  # Learned policy
# agent.policy_net.load_state_dict(torch.load('model.pth'))
```

---

## ✅ 장점

1. **최소 수정**: 기존 코드 3곳만 수정
2. **점진적 적용**: 규칙 → RL 단계적 전환
3. **모듈화**: RL 로직 분리, 재사용 가능
4. **호환성**: 기존 코드 그대로 실행 가능
5. **테스트 용이**: 여러 모드 비교 쉬움

---

## 🔍 디버깅

### RL Agent 직접 테스트
```bash
python test_rl_integration.py --test agent
```

### 시뮬레이션만 테스트
```bash
python test_rl_integration.py --test sim --flights 5
```

### RL 결정 로그 확인
```python
# rl_agent.py에 로깅 추가
def decide_charging(unit, chargers, time):
    action = agent.select_charging_action(state)
    print(f"[RL] AGV {unit.global_id}: battery={state.battery:.2f} → action={action}")
    return charger_name
```

---

## 📌 다음 단계

1. ✅ 기본 통합 완료 (현재)
2. ⬜ 실제 DQN/PPO agent 구현
3. ⬜ Replay buffer 및 학습 루프
4. ⬜ Reward shaping 튜닝
5. ⬜ 성능 비교 (Rule vs RL)

---

## 💡 핵심 요약

**"기존 코드의 2가지 decision point (`amr_task_end`, `flight_arrival`)에서 `rl_agent` 모듈만 호출하면 끝!"**

- 충전 결정: `rl_agent.decide_charging()`
- 비딩 결정: `rl_agent.select_units_by_rl()`

간단하죠? 🎉
