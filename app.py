import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# 1. 한글 폰트 설정 (기본 폰트 사용으로 에러 방지)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 2. 페이지 및 다크 스타일 설정
st.set_page_config(page_title="온도 제어 시뮬레이터", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stSelectbox label, .stNumberInput label, .stCheckbox label {
        color: #e0e0e0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. 사이드바 입력 설정
st.sidebar.header("⚙️ 공정 변수")
SP = st.sidebar.number_input("목표 온도 (Set Point, °C)", value=1000.0, step=10.0)
delay_time = st.sidebar.number_input("시간 지연 (L, 초)", value=30, step=1)
total_time = st.sidebar.number_input("총 시뮬레이션 시간 (초)", value=5400, step=300)
dt = st.sidebar.number_input("시간 간격 (초)", value=1, step=1)

st.sidebar.header("🎛️ 제어기 설정")
onoff_hyst = st.sidebar.number_input("온오프 히스테리시스 (±°C)", value=5.0, step=0.5)
Kc_user = st.sidebar.number_input("P/PID 비례 이득 (Kc)", value=0.30, step=0.05)
tau_I_user = st.sidebar.number_input("PID 적분 시간 (τI, 초)", value=200.0, step=10.0)
tau_D_user = st.sidebar.number_input("PID 미분 시간 (τD, 초)", value=25.0, step=5.0)

st.sidebar.header("📊 그래프 표시 설정")
show_onoff = st.sidebar.checkbox("온오프 온도", value=True)
show_p = st.sidebar.checkbox("P 온도", value=True)
show_pid = st.sidebar.checkbox("PID 온도", value=True)
show_sp = st.sidebar.checkbox("설정값", value=True)

# 4. 시뮬레이션 로직
n_steps = int(total_time / dt) + 1
t = np.linspace(0, total_time, n_steps)
T0 = 25.0
delay_steps = int(delay_time / dt)

def run_sim(c_type, Kc=0.3, tau_I=200.0, tau_D=25.0):
    T = np.zeros(n_steps)
    u = np.zeros(n_steps)
    T[0] = T0
    integral = 0.0
    
    for k in range(n_steps):
        curr_T = T[k]
        if c_type == 'on_off':
            if k == 0:
                u_val = 100.0 if curr_T < (SP - onoff_hyst) else 0.0
            else:
                if curr_T < (SP - onoff_hyst):
                    u_val = 100.0
                elif curr_T > (SP + onoff_hyst):
                    u_val = 0.0
                else:
                    u_val = u[k-1]
            u[k] = u_val
        elif c_type == 'P':
            e = SP - curr_T
            u[k] = np.clip(Kc * e, 0.0, 100.0)
        elif c_type == 'PID':
            e = SP - curr_T
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
            u_del = u[k - delay_steps] if k >= delay_steps else 0.0
            dT_dt = (-(curr_T - T0) + 16.0 * u_del) / 300.0
            T[k+1] = curr_T + dT_dt * dt
    return T, u

T_onoff, u_onoff = run_sim('on_off')
T_p, u_p = run_sim('P', Kc=Kc_user)
T_pid, u_pid = run_sim('PID', Kc=Kc_user, tau_I=tau_I_user, tau_D=tau_D_user)

# 5. 화면 구성 (타이틀 및 탭)
st.title("온도 제어 시뮬레이터")
st.caption("FOPDT 공정에서 On-Off, P, PID 제어를 비교합니다.")

tab1, tab2 = st.tabs(["📈 그래프 화면", "📋 수치표"])

with tab1:
    col1, col2 = st.columns(2)
    
    # [1] On-Off 전체
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 3.8), facecolor='#0e1117')
        ax1.set_facecolor('#0e1117')
        if show_onoff:
            ax1.plot(t/60, T_onoff, color='#1f77b4', label='On-Off')
        if show_sp:
            ax1.axhline(SP, color='red', linestyle='--', label='Set Point')
        ax1.set_title("1. On-Off Control", color='white')
        ax1.set_xlabel("Time (min)", color='white')
        ax1.set_ylabel("Temp (C)", color='white')
        ax1.tick_params(colors='white')
        ax1.grid(True, color='#333333', linestyle=':')
        ax1.legend(facecolor='#1e222a', edgecolor='none', labelcolor='white')
        st.pyplot(fig1)

    # [2] On-Off 확대 (40~70분)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 3.8), facecolor='#0e1117')
        ax2.set_facecolor('#0e1117')
        idx1, idx2 = int(2400/dt), int(4200/dt)
        if show_onoff:
            ax2.plot(t[idx1:idx2]/60, T_onoff[idx1:idx2], color='#1f77b4')
        ax2.set_title("2. On-Off Zoom (40~70 min)", color='white')
        ax2.set_xlabel("Time (min)", color='white')
        ax2.set_ylabel("Temp (C)", color='white')
        ax2.tick_params(colors='white')
        ax2.grid(True, color='#333333', linestyle=':')
        st.pyplot(fig2)

    # [3] 세 제어기 온도 비교
    fig3, ax3 = plt.subplots(figsize=(12, 4.5), facecolor='#0e1117')
    ax3.set_facecolor('#0e1117')
    if show_onoff:
        ax3.plot(t/60, T_onoff, label='On-Off', color='#1f77b4', alpha=0.6)
    if show_p:
        ax3.plot(t/60, T_p, label='P Control', color='#ff7f0e')
    if show_pid:
        ax3.plot(t/60, T_pid, label='PID Control', color='#2ca02c')
    if show_sp:
        ax3.axhline(SP, color='white', linestyle='--', label='Set Point')
    
    ax3.set_title("3. Controller Comparison", color='white')
    ax3.set_xlabel("Time (min)", color='white')
    ax3.set_ylabel("Temp (C)", color='white')
    ax3.set_ylim(750, 1100)
    ax3.tick_params(colors='white')
    ax3.grid(True, color='#333333', linestyle=':')
    ax3.legend(facecolor='#1e222a', edgecolor='none', labelcolor='white')
    st.pyplot(fig3)

with tab2:
    st.subheader("시간별 시뮬레이션 수치")
    df = pd.DataFrame({
        'Time(s)': t,
        'SP': SP,
        'OnOff_Temp(C)': T_onoff,
        'OnOff_Heater(%)': u_onoff,
        'P_Temp(C)': T_p,
        'P_Heater(%)': u_p,
        'PID_Temp(C)': T_pid,
        'PID_Heater(%)': u_pid
    })
    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 CSV 결과 파일 다운로드",
        data=csv_data,
        file_name='simulation_results.csv',
        mime='text/csv'
    )
