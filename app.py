import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Smart Money Portfolio", layout="wide")

# ==========================================
# 1. PORTFOLIO MANAGEMENT (JSON STORAGE)
# ==========================================
PORTFOLIO_FILE = 'portfolio.json'

def load_portfolio():
    """โหลดข้อมูลพอร์ตจากไฟล์ JSON"""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_portfolio(data):
    """บันทึกข้อมูลพอร์ตลงไฟล์ JSON"""
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(data, f)

# ==========================================
# 2. HMM LOGIC CLASS
# ==========================================
class SmartMoneyHMM:
    def __init__(self, ticker, period='1y', interval='1d', n_states=4):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.n_states = n_states
        self.df = None
        self.model = None
        self.accum_state_id = None
        self.accum_stats = {}
        self.state_props = {}

    def fetch_data(self):
        try:
            self.df = yf.download(self.ticker, period=self.period, interval=self.interval, progress=False)
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = self.df.columns.get_level_values(0)
            
            if self.df.empty: return False
            self.df = self.df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            self.df = self.df[self.df['Volume'] > 0]
            return True
        except Exception:
            return False

    def add_indicators(self):
        df = self.df.copy()
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        window = 20
        sma = df['Close'].rolling(window).mean()
        std = df['Close'].rolling(window).std()
        df['BB_Width'] = (4 * std) / sma
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Rel_Vol'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, 1)
        self.df = df.dropna()

    def train_hmm(self):
        feature_cols = ['Log_Ret', 'BB_Width', 'RSI', 'Rel_Vol']
        X = self.df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.model = GaussianHMM(n_components=self.n_states, covariance_type='full', n_iter=1000, random_state=42)
        self.model.fit(X_scaled)
        self.df['HMM_State'] = self.model.predict(X_scaled)

    def interpret_states(self):
        state_stats = {}
        for i in range(self.n_states):
            mask = self.df['HMM_State'] == i
            if not mask.any(): continue
            state_stats[i] = {
                'volatility': self.df.loc[mask, 'BB_Width'].mean(),
                'return': self.df.loc[mask, 'Log_Ret'].mean()
            }
        
        self.state_props = {} 
        if self.n_states == 4:
            # Logic: Volatility ต่ำสุด = Accumulation
            accum_id = min(state_stats, key=lambda k: state_stats[k]['volatility'])
            self.state_props[accum_id] = {'color': '#10B981', 'label': 'Accumulation (เก็บของ)'}
            
            remaining = [k for k in state_stats if k != accum_id]
            
            # Logic: Return ต่ำสุด = Markdown
            markdown_id = min(remaining, key=lambda k: state_stats[k]['return'])
            self.state_props[markdown_id] = {'color': '#EF4444', 'label': 'Markdown (ขาลง)'}
            
            remaining = [k for k in remaining if k != markdown_id]
            
            # Logic: Volatility สูงสุดที่เหลือ = Distribution
            dist_id = max(remaining, key=lambda k: state_stats[k]['volatility'])
            self.state_props[dist_id] = {'color': '#F97316', 'label': 'Distribution (ระบายของ)'}
            
            remaining = [k for k in remaining if k != dist_id]
            
            if remaining:
                self.state_props[remaining[0]] = {'color': '#3B82F6', 'label': 'Markup (ขาขึ้น)'}
            
            self.accum_state_id = accum_id
        else:
            # Fallback for != 4 states
            accum_id = min(state_stats, key=lambda k: state_stats[k]['volatility'])
            self.accum_state_id = accum_id
            colors = ['#10B981', '#3B82F6', '#F97316', '#EF4444', '#8B5CF6']
            for i in state_stats:
                color = colors[i % len(colors)]
                label = 'Accumulation' if i == accum_id else f'State {i}'
                self.state_props[i] = {'color': color, 'label': label}

        # VWAP Stats
        accum_data = self.df[self.df['HMM_State'] == self.accum_state_id]
        if not accum_data.empty:
            vwap = (accum_data['Close'] * accum_data['Volume']).sum() / accum_data['Volume'].sum()
            self.accum_stats = {'vwap': vwap}

# ==========================================
# 3. MAIN APPLICATION LOGIC
# ==========================================

# Initialize Session State
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = "BTC-USD"
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_portfolio()

def update_ticker(symbol):
    st.session_state.current_ticker = symbol

# --- SIDEBAR: PORTFOLIO MANAGER ---
with st.sidebar:
    st.title("💼 Portfolio")
    
    # 1. Add New Stock Form
    with st.expander("➕ เพิ่มหุ้นเข้าพอร์ต", expanded=False):
        new_ticker = st.text_input("ชื่อหุ้น (Symbol)", placeholder="e.g. AAPL").strip().upper()
        new_cost = st.number_input("ต้นทุนเฉลี่ย (USD)", min_value=0.0, format="%.2f")
        new_qty = st.number_input("จำนวนที่ถือ (Qty)", min_value=0.0, format="%.4f")
        
        if st.button("บันทึก"):
            if new_ticker:
                st.session_state.portfolio[new_ticker] = {
                    'cost': new_cost,
                    'qty': new_qty
                }
                save_portfolio(st.session_state.portfolio)
                st.success(f"บันทึก {new_ticker} แล้ว")
                st.rerun()
            else:
                st.error("กรุณาใส่ชื่อหุ้น")

    st.divider()

    # 2. List Saved Stocks
    st.subheader("รายการหุ้นที่บันทึกไว้")
    if not st.session_state.portfolio:
        st.info("ยังไม่มีหุ้นในพอร์ต")
    else:
        # วนลูปแสดงรายชื่อหุ้น
        for tick, data in list(st.session_state.portfolio.items()):
            col_btn, col_del = st.columns([3, 1])
            
            # ปุ่มกดเพื่อดู analysis
            with col_btn:
                if st.button(f"🔍 {tick}", key=f"btn_{tick}", use_container_width=True):
                    update_ticker(tick)
            
            # ปุ่มลบ
            with col_del:
                if st.button("🗑️", key=f"del_{tick}"):
                    del st.session_state.portfolio[tick]
                    save_portfolio(st.session_state.portfolio)
                    st.rerun()

# --- MAIN CONTENT ---
st.title(f"📊 Analysis: {st.session_state.current_ticker}")

# Input for temporary check (Manual search)
manual_ticker = st.text_input("ค้นหาหุ้นอื่นชั่วคราว (ไม่บันทึก)", value=st.session_state.current_ticker)
if manual_ticker.upper() != st.session_state.current_ticker:
    st.session_state.current_ticker = manual_ticker.upper()

# Settings
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1: period = st.selectbox("Period", ['6mo', '1y', '2y'], index=1)
with col_s2: interval = st.selectbox("Timeframe", ['1d', '1wk'], index=0)
with col_s3: n_states = 4

# Run Analysis
model = SmartMoneyHMM(st.session_state.current_ticker, period, interval, n_states)
with st.spinner("กำลังวิเคราะห์ข้อมูล..."):
    if model.fetch_data():
        model.add_indicators()
        model.train_hmm()
        model.interpret_states()

        df = model.df
        last_price = df['Close'].iloc[-1]
        accum_price = model.accum_stats.get('vwap', 0)
        
        # --- SECTION: PORTFOLIO PERFORMANCE (ถ้าหุ้นนี้อยู่ในพอร์ต) ---
        if st.session_state.current_ticker in st.session_state.portfolio:
            port_data = st.session_state.portfolio[st.session_state.current_ticker]
            my_cost = port_data['cost']
            my_qty = port_data['qty']
            
            market_value = last_price * my_qty
            total_cost = my_cost * my_qty
            unrealized_pl = market_value - total_cost
            pl_percent = ((last_price - my_cost) / my_cost * 100) if my_cost > 0 else 0
            
            st.markdown("### 💰 สถานะพอร์ตของคุณ")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("จำนวนที่ถือ", f"{my_qty:,.4f}")
            p2.metric("ต้นทุนเฉลี่ย", f"${my_cost:,.2f}")
            p3.metric("มูลค่าปัจจุบัน", f"${market_value:,.2f}")
            p4.metric("กำไร/ขาดทุน (P/L)", f"${unrealized_pl:,.2f}", f"{pl_percent:+.2f}%", 
                      delta_color="normal")
            
            st.divider()

        # --- SECTION: AI ANALYSIS ---
        st.markdown("### 🤖 AI Smart Money Analysis")
        m1, m2, m3 = st.columns(3)
        m1.metric("ราคาตลาด", f"${last_price:,.2f}")
        
        # เทียบราคาตลาด กับ ต้นทุนเจ้ามือ
        sm_gap = ((last_price - accum_price) / accum_price * 100) if accum_price else 0
        m2.metric("ต้นทุนเจ้ามือ (Accum VWAP)", f"${accum_price:,.2f}", f"{sm_gap:+.2f}% vs Market")
        
        current_state_color = model.state_props[df['HMM_State'].iloc[-1]]['color']
        current_state_label = model.state_props[df['HMM_State'].iloc[-1]]['label']
        m3.markdown(f"**สถานะ:** <span style='color:{current_state_color};font-weight:bold;font-size:1.2em'>{current_state_label}</span>", unsafe_allow_html=True)

        # Plot Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)

        # 1. Price
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price', line=dict(color='gray', width=1), opacity=0.3), row=1, col=1)
        
        # 2. Portfolio Cost Line (ถ้ามี)
        if st.session_state.current_ticker in st.session_state.portfolio:
             fig.add_hline(y=st.session_state.portfolio[st.session_state.current_ticker]['cost'], 
                           line_dash="dash", line_color="yellow", annotation_text="My Cost", row=1, col=1)

        # 3. Colored Dots
        sorted_states = sorted(model.state_props.keys(), key=lambda x: 0 if x == model.accum_state_id else 1)
        for state_id in sorted_states:
            mask = df['HMM_State'] == state_id
            props = model.state_props[state_id]
            fig.add_trace(go.Scatter(
                x=df.index[mask], y=df['Close'][mask], mode='markers',
                name=props['label'], marker=dict(color=props['color'], size=5), opacity=0.9
            ), row=1, col=1)

        # 4. RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#A78BFA', width=1)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

        fig.update_layout(height=600, template="plotly_dark", hovermode="x unified", margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error(f"ไม่พบข้อมูลหุ้น {st.session_state.current_ticker} หรือชื่อผิด")

