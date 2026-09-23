import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# 페이지 기본 설정
st.set_page_config(page_title="PID 제어 시뮬레이터", layout="wide")

# 타이틀
st.title("🔥 온오프 / P / PID 제어 시뮬레이션 웹앱")
st.markdown("왼쪽 사이드바에서 **PID 제어 파라미터**를 조절하면 오른쪽 그래프가 **실시간으로 업데이트**됩니다.")

# ==========================================
# 1. 사이드바 - 파라미터 입력창
# ==========================================
st.sidebar.header("⚙️ PID 제어기 설정")

Kc_user = st.sidebar.number_input("1) Kc (비례 이득)", min_value=0.0, max_value=10.0, value=0.3, step=0.1)
tau_I_user = st.sidebar.number_input("2) τI (적분 시간, 초)", min_value=1.0, max_value=2000.0, value=200.0, step=10.0)
tau_D_user = st.sidebar.number_input("3) τD (미분 시간, 초)", min_value=0.0, max_value=500.0, value=25.0, step=5.0)

# ==========================================
# 2. 시뮬레이션 설정 및 제어 로직
# ==========================================
dt = 1.0               # 시간 간격 Δt = 1초
total_time = 5400      # 총 시뮬레이션 시간 = 5400초 (90분)
n_steps = int(total_time / dt) + 1
t = np.linspace(0, total_time, n_steps)

SP = 1000.0            # 목표 설정값 (°C)
T0 = 25.0              # 초기 상온 (°C)
delay_steps = 30       # 시간 지연 L = 30초

def run_simulation(controller_type, Kc=0.3, tau_I=200.0, tau_D=25.0):
    T = np.zeros(n_steps)
    u = np.zeros(n_steps)
    T[0] = T0
    integral = 0.0
    
    for k in range(n_steps):
        current_T = T[k]
        
        if controller_type == 'on_off':
            if k == 0:
                u_val = 100.0 if current_T < 995.0 else 0.0
            else:
                if current_T < 995.0:
                    u_val = 100.0
                elif current_T > 1005.0:
                    u_val = 0.0
                else:
                    u_val = u[k-1]
            u[k] = u_val
            
        elif controller_type == 'P':
            e = SP - current_T
            u_raw = 0.3 * e
            u[k] = np.clip(u_raw, 0.0, 100.0)
            
        elif controller_type == 'PID':
            e = SP - current_T
            e_prev = (SP - T[k-1]) if k > 0 else e
            de = (e - e_prev) / dt
            
            P_term = Kc * e
            I_term = integral
            D_term = Kc * tau_D * de
            
            u_raw = P_term + I_term + D_term
            
            if 0.0 <= u_raw <= 100.0 and tau_I > 0:
                integral += (Kc / tau_I) * e * dt
                
            u[k] = np.clip(u_raw, 0.0, 100.0)

        if k < n_steps - 1:
            u_delayed = u[k - delay_steps] if k >= delay_steps else 0.0
            dT_dt = (-(current_T - 25.0) + 16.0 * u_delayed) / 300.0
            T[k+1] = current_T + dT_dt * dt
            
    return T, u

# 시뮬레이션 계산 실행
T_onoff, u_onoff = run_simulation('on_off')
T_p, u_p = run_simulation('P')
T_pid, u_pid = run_simulation('PID', Kc=Kc_user, tau_I=tau_I_user, tau_D=tau_D_user)

# 분석 결과 산출
last_30min_idx = int(3600 / dt)
onoff_oscillation = np.max(T_onoff[last_30min_idx:]) - np.min(T_onoff[last_30min_idx:])
p_offset = SP - T_p[-1]
pid_error = SP - T_pid[-1]

# ==========================================
# 3. 화면 구성 및 결과 출력
# ==========================================
col1, col2, col3 = st.columns(3)
col1.metric("On-Off 진동 폭 (최근 30분)", f"{onoff_oscillation:.2f} °C")
col2.metric("P 제어 잔류편차 (Offset)", f"{p_offset:.2f} °C")
col3.metric("PID 제어 최종 오차", f"{pid_error:.2f} °C")

# 그래프 시각화
fig = plt.figure(figsize=(10, 10))

# [1] 온오프 전체
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(t, T_onoff, color='tab:blue', label='On-Off Temp T(t)')
ax1.axhline(SP, color='red', linestyle='--', label='Set Point (1000°C)')
ax1.set_title('1) On-Off Control - Full Temp Curve')
ax1.set_ylabel('Temp (°C)')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

# [2] 40~70분 확대
ax2_temp = fig.add_subplot(3, 1, 2)
idx_40min, idx_70min = int(2400 / dt), int(4200 / dt)
t_sub = t[idx_40min:idx_70min]

ax2_temp.plot(t_sub, T_onoff[idx_40min:idx_70min], color='tab:blue', label='Temp T(t)')
ax2_temp.set_title('2) On-Off Control - Zoom 40~70 min')
ax2_temp.set_ylabel('Temp (°C)', color='tab:blue')
ax2_temp.grid(True, linestyle=':', alpha=0.7)

ax2_heat = ax2_temp.twinx()
ax2_heat.plot(t_sub, u_onoff[idx_40min:idx_70min], color='tab:red', drawstyle='steps-post', alpha=0.7, label='Heater u(t)')
ax2_heat.set_ylabel('Heater Output (%)', color='tab:red')
ax2_heat.set_ylim(-10, 110)

# [3] 세 제어기 비교
ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(t, T_onoff, label='On-Off', color='tab:blue', alpha=0.5)
ax3.plot(t, T_p, label='P Control', color='tab:orange')
ax3.plot(t, T_pid, label=f'PID (Kc={Kc_user}, Tau_I={tau_I_user}, Tau_D={tau_D_user})', color='tab:green')
ax3.axhline(SP, color='red', linestyle='--')
ax3.set_title('3) Controller Comparison (750~1100°C)')
ax3.set_xlabel('Time (sec)')
ax3.set_ylabel('Temp (°C)')
ax3.set_ylim(750, 1100)
ax3.grid(True, linestyle=':', alpha=0.7)
ax3.legend()

plt.tight_layout()
st.pyplot(fig)

# ==========================================
# 4. 데이터 내려받기 (CSV 다운로드)
# ==========================================
st.subheader("📥 시뮬레이션 결과 데이터 다운로드")

df = pd.DataFrame({
    'Time(s)': t,
    'OnOff_Temp(C)': T_onoff,
    'OnOff_Heater(%)': u_onoff,
    'P_Temp(C)': T_p,
    'P_Heater(%)': u_p,
    'PID_Temp(C)': T_pid,
    'PID_Heater(%)': u_pid
})

csv_data = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📄 시뮬레이션 결과 CSV 파일 다운로드",
    data=csv_data,
    file_name='simulation_results.csv',
    mime='text/csv'
)
