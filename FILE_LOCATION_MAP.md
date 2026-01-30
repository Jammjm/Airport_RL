# 핵심 요약 내용의 파일 위치 맵

## 📍 주요 내용별 파일 위치

### 1️⃣ STATE 정의 (상태 정보 구조)
**파일**: `reporting.py`
- **함수**: `update_state()` (line 315)
- **내용**: 
  - AMR 상태 수집: `global_id`, `kind`, `soc_percent`, `location_label`, `state`, `total_work_time`
  - Charger 상태 수집: 큐 길이
  - snapshot 생성 및 LOG에 저장

**또한**: `reporting_original.py`에도 동일한 함수 (line 170) - 원본 파일

---

### 2️⃣ ACTION 정의 (Dispatch Rule)
**파일**: `model.py`
- **함수**: `_select_units_by_rule()` (line 214)
- **내용**:
  - FIFO 규칙
  - RANDOM 규칙
  - LEAST_UTILIZED 규칙
  - BIDDING 규칙 (battery SOC + utilization 기반)

**설정**: `config.py` (line 23)
- `DISPATCHING_RULE = 'RANDOM'` ← 현재 선택된 규칙

---

### 3️⃣ REQUEST-RELEASE 사이클
**파일**: `model.py`

#### REQUEST (유닛 할당)
- **함수**: `request_units()` (line 246)
- **내용**: 
  - 사용 가능한 유닛 찾기 (`_get_eligible_units()`)
  - dispatch rule 적용 (`_select_units_by_rule()`)
  - 유닛을 task에 할당

#### RELEASE (유닛 반환)
- **함수**: `release_units()` (line 273)
- **내용**:
  - Task 완료 후 state update ("amr_task_end" trigger)
  - 필요시 charging
  - Depot으로 복귀
  - Available 상태로 변경

---

### 4️⃣ 시간 전파 (Time Propagation)
**파일**: `model.py`

#### Travel (이동)
- **함수**: `_travel()` (line 102)
- **코드**: 
  ```python
  unit.consume_energy(travel_time, cfg.TRAVEL_CONSUME_POWER_KW, self.kpi)
  yield self.env.timeout(travel_time)  # ← 시간 진행
  ```

#### Service (서비스)
- **함수**: `_service()` (line 130)
- **코드**:
  ```python
  unit.consume_energy(duration_min, cfg.DEFAULT_SERVICE_CONSUME_POWER_KW, self.kpi)
  yield self.env.timeout(duration_min)  # ← 시간 진행
  ```

#### Charging (충전)
- **함수**: `_charge()` (line 145)
- **코드**:
  ```python
  hours_to_charge = need_kwh / CHARGE_POWER_KW
  duration_min = hours_to_charge * 60.0
  yield self.env.timeout(duration_min)  # ← 시간 진행
  unit.soc_kwh = unit.capacity_kwh      # ← SoC 회복
  ```

---

### 5️⃣ 배터리 SoC 전파 (Battery Propagation)
**파일**: `model.py`

#### Energy Consumption
- **함수**: `consume_energy()` (line 50, AMRUnit 클래스)
- **내용**:
  ```python
  def consume_energy(self, duration_min: float, power_kw: float, kpi):
      used_kwh = power_kw * (duration_min / 60.0)
      self.soc_kwh = max(0.0, self.soc_kwh - used_kwh)  # ← 즉시 감소
      kpi.total_energy_consumed += used_kwh
  ```
- **호출**: 
  - Travel 중 (line 118)
  - Service 중 (line 136)
  - GPU service 중 (line 469)

#### Energy Charging
- **파일**: `model.py`, `_charge()` 함수
- **코드**: 
  ```python
  unit.soc_kwh = unit.capacity_kwh  # ← SOC 100%로 회복
  kpi.total_charge_kwh += need_kwh
  ```

---

### 6️⃣ STATE UPDATE 트리거 포인트
**파일**: `model.py`

#### Flight Arrival 시점
- **함수**: `flight_process()` (line 388)
- **코드**: `update_state(env, "flight_arrival", kpi, fleets)` (line 404)
- **역할**: 비행기가 gate에 도착했을 때 state snapshot 생성

#### Task Completion 시점
- **함수**: `_unit_return_logic()` 내부 (line 280)
- **코드**: `update_state(self.env, "amr_task_end", self.kpi, all_fleets)`
- **역할**: AMR이 task를 완료했을 때 state snapshot 생성

---

### 7️⃣ Energy & Charging 관련 설정
**파일**: `config.py`

```python
# Battery Capacity
DEFAULT_BATTERY_CAP_KWH = 40.0  # 일반 AMR
GPU_CONFIG = {'BATTERY_CAP_KWH': 150.0, ...}  # GPU AMR

# Energy Consumption
TRAVEL_CONSUME_POWER_KW = 24.4  # 이동 중 소비
DEFAULT_SERVICE_CONSUME_POWER_KW = 10.0  # 서비스 중 소비
GPU_CONFIG['SERVICE_CONSUME_POWER_KW'] = 30.0  # GPU 서비스 중 소비

# Charging
CHARGE_TRIGGER_SOC = 0.3  # 30% 이하면 충전 시작
CHARGE_POWER_KW = 12.2  # 충전 속도
CHARGER_CAPACITY = 3  # 동시 충전 가능 수
```

---

### 8️⃣ 비행기 프로세스
**파일**: `model.py`

#### Main Flight Process
- **함수**: `flight_process()` (line 388)
- **단계**:
  1. Gate 할당 대기
  2. Flight arrival state update
  3. GPU & OTHER tasks 병렬 시작
  4. GPU process 실행
  5. OTHER tasks 완료 대기
  6. GPU unit 반환
  7. Gate 해제

#### Flight Starter
- **함수**: `flight_starter()` (line 485)
- **역할**: 특정 시간에 flight_process 스케줄링

---

### 9️⃣ KPI & Reporting
**파일**: `reporting.py`

- **클래스**: `KPIs` (line 38)
- **추적 항목**:
  - Flight turnaround time
  - Flight delays
  - Gate wait times
  - GPU arrival wait times
  - Total travel distance
  - Total energy consumed
  - AMR utilization (시간 기반)
  - Charger utilization

---

## 📊 파일 구조 요약

```
config.py
├─ 모든 설정 변수 정의
└─ DISPATCHING_RULE, 에너지 설정, 충전 설정

routing.py
├─ 지도 좌표 (NODE_POS, GATE_LABELS)
├─ 경로 계산 (_get_path_waypoints)
└─ 거리/시간 계산 (_calculate_path_distance_and_time)

model.py (← 핵심 시뮬레이션 로직)
├─ AMRUnit 클래스 (consume_energy)
├─ ChargerBank 클래스 (충전소)
├─ AMRFleet 클래스
│  ├─ _travel() → time & energy propagate
│  ├─ _service() → time & energy propagate
│  ├─ _charge() → time & energy propagate
│  ├─ request_units() → dispatch rule 적용
│  ├─ release_units() → state update 트리거
│  └─ _select_units_by_rule() → ACTION 선택
├─ _task_process() (각 task 실행)
├─ flight_process() → state update ("flight_arrival")
└─ flight_starter()

reporting.py (← 상태 관리 & KPI)
├─ EventLogger 클래스 (event logging)
├─ KPIs 클래스 (KPI 추적)
├─ update_state() → state snapshot 생성
├─ _setup_output_dir()
├─ _export_logs()
└─ _plot_gate_gantt()

main.py
└─ 시뮬레이션 메인 루틴 (위 모든 모듈 통합)
```

---

## 🎯 RL 통합을 위한 수정 위치

### 1. State 받기
- **파일**: `reporting.py`의 `update_state()` 함수 내
- **위치**: snapshot 생성 후 (line 345~)

### 2. Action 주기
- **파일**: `model.py`의 `_select_units_by_rule()` 함수
- **위치**: dispatch rule 선택 부분 (line 214~243)

### 3. Reward 계산
- **파일**: `model.py`의 `_unit_return_logic()` 또는 `flight_process()` 내
- **위치**: task completion 후 state update 직후

---

## 참고: 원본 vs 분할 버전

- **Whole_SIM.py**: 원본 (모두 한 파일)
- **분할 버전**: 
  - config.py
  - routing.py
  - model.py
  - reporting.py
  - main.py

두 버전 모두 동일한 로직을 포함하고 있습니다.
