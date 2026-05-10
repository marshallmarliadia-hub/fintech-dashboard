import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import joblib
import re
import os
from collections import Counter

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Fintech Insight Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0F1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * { color: #c9cdd6 !important; }
[data-testid="stSidebar"] .stRadio label { 
    font-size: 14px; padding: 6px 0; 
}

/* Header */
.dash-header {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
    border: 1px solid #1e2130;
}
.dash-title {
    font-family: 'DM Serif Display', serif;
    font-size: 26px;
    color: #f0f2f8;
    margin: 0 0 6px;
    line-height: 1.2;
}
.dash-sub {
    font-size: 13px;
    color: #6b7385;
    margin: 0;
    letter-spacing: 0.02em;
}

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.metric-card {
    background: #1a1f2e;
    border: 1px solid #252b3b;
    border-radius: 10px;
    padding: 16px 20px;
    flex: 1; min-width: 140px;
}
.metric-label { font-size: 11px; color: #6b7385; letter-spacing: 0.06em; text-transform: uppercase; margin: 0 0 6px; }
.metric-value { font-size: 26px; font-weight: 500; color: #f0f2f8; margin: 0; line-height: 1; }
.metric-sub   { font-size: 11px; color: #4e5568; margin: 4px 0 0; }

/* Section title */
.section-title {
    font-size: 13px;
    font-weight: 500;
    color: #6b7385;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0 0 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2130;
}

/* Insight box */
.insight-box {
    background: #1a1f2e;
    border-left: 3px solid #4f6ef7;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 13px;
    color: #9ca3b8;
    line-height: 1.7;
}
.insight-box strong { color: #c9cdd6; }

/* Badge */
.badge {
    display: inline-block;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 500;
    margin: 2px 3px;
}
.badge-green  { background:#0d2e1f; color:#34d399; }
.badge-red    { background:#2e0d0d; color:#f87171; }
.badge-blue   { background:#0d1a2e; color:#60a5fa; }
.badge-yellow { background:#2e2100; color:#fbbf24; }

/* Prediction panel */
.pred-result {
    background: #1a1f2e;
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #252b3b;
    text-align: center;
}
.pred-label { font-size: 13px; color: #6b7385; margin-bottom: 8px; }
.pred-value { font-size: 32px; font-weight: 500; margin: 0; }
.pred-positive { color: #34d399; }
.pred-negative { color: #f87171; }
.pred-neutral  { color: #fbbf24; }

/* Divider */
.hdivider { border: none; border-top: 1px solid #1e2130; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

@st.cache_data
def load_reviews():
    path = "reviews_preprocessed.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    # fallback dummy
    return pd.DataFrame({
        'sentiment': np.random.choice(['positive','negative','neutral'],
                                       size=500, p=[0.727,0.239,0.034]),
        'stars': np.random.choice([1,2,3,4,5], size=500),
        'review_body': ['sample review']*500,
        'review_clean': ['sample review']*500,
        'country': np.random.choice(['US','UK','DE'], size=500),
        'year': np.random.choice([2020,2021,2022], size=500),
        'month': np.random.choice(range(1,13), size=500),
    })

@st.cache_data(ttl=60)
def load_fraud():
    for path in ["fraud_cleaned.csv", "fraud_cleaned (1).csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df, False   # (dataframe, is_dummy)
    # fallback dummy — hanya jika tidak ada file sama sekali
    np.random.seed(42)
    n = 5000
    types = np.random.choice(['CASH_OUT','PAYMENT','CASH_IN','TRANSFER','DEBIT'], n,
                              p=[0.356,0.337,0.217,0.083,0.007])
    fraud = np.random.choice([0,1], n, p=[0.999,0.001])
    df_dummy = pd.DataFrame({
        'transaction_type': types,
        'amount': np.random.exponential(150000, n),
        'is_fraud': fraud,
        'fraud_label': np.where(fraud==1,'fraud','legitimate'),
        'day': np.random.choice(range(1,5), n),
        'hour_of_day': np.random.choice(range(1,25), n),
        'balance_change_orig': np.random.normal(0, 100000, n),
        'zero_orig_balance': np.random.choice([0,1], n, p=[0.7,0.3]),
    })
    return df_dummy, True   # (dataframe, is_dummy)

@st.cache_resource
def load_model_reviews():
    for p in ['best_model_reviews.pkl','model_reviews.pkl']:
        if os.path.exists(p):
            return joblib.load(p)
    return None

@st.cache_resource
def load_model_fraud():
    for p in ['best_model_fraud.pkl','model_fraud.pkl']:
        if os.path.exists(p):
            return joblib.load(p)
    return None

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    stopwords = {'the','a','an','is','it','in','on','at','to','for','of',
                 'and','or','but','i','my','we','you','was','are','this',
                 'that','with','have','has','had','be','been','not','no',
                 'its','their','they','he','she','me','him','her','us','do',
                 'did','does','will','would','could','should','from','by'}
    tokens = [w for w in text.split() if w not in stopwords and len(w)>2]
    return ' '.join(tokens)

PALETTE = {
    'positive': '#34d399',
    'negative': '#f87171',
    'neutral':  '#fbbf24',
    'fraud':    '#f87171',
    'legit':    '#34d399',
    'accent':   '#4f6ef7',
    'muted':    '#252b3b',
    'bg':       '#1a1f2e',
    'text':     '#c9cdd6',
    'grid':     '#1e2130',
}

def dark_fig(figsize=(10,4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0f1117')
    ax.set_facecolor('#0f1117')
    for spine in ax.spines.values():
        spine.set_color('#1e2130')
    ax.tick_params(colors='#6b7385', labelsize=9)
    ax.xaxis.label.set_color('#6b7385')
    ax.yaxis.label.set_color('#6b7385')
    ax.title.set_color('#c9cdd6')
    ax.grid(color='#1e2130', linewidth=0.5, alpha=0.8)
    return fig, ax

def dark_figs(nrows, ncols, figsize):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='#0f1117')
    fig.patch.set_facecolor('#0f1117')
    for ax in (axes.flat if hasattr(axes,'flat') else [axes]):
        ax.set_facecolor('#0f1117')
        for spine in ax.spines.values():
            spine.set_color('#1e2130')
        ax.tick_params(colors='#6b7385', labelsize=9)
        ax.grid(color='#1e2130', linewidth=0.5, alpha=0.8)
        ax.xaxis.label.set_color('#6b7385')
        ax.yaxis.label.set_color('#6b7385')
    return fig, axes


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📊 Fintech Dashboard")
    st.markdown("<hr style='border-color:#1e2130;margin:8px 0 16px'>", unsafe_allow_html=True)
    page = st.radio("Navigasi", [
        "🏠  Ringkasan",
        "💬  Reviews & Sentimen",
        "🔍  Fraud Detection",
        "🔗  Integrasi Insight",
        "🤖  Prediksi Real-time",
    ])
    st.markdown("<hr style='border-color:#1e2130;margin:16px 0 12px'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px;color:#4e5568;line-height:1.8'>
    <b style='color:#6b7385'>Proyek</b><br>
    Dari Persepsi ke Transaksi<br><br>
    <b style='color:#6b7385'>Referensi</b><br>
    Carbo-Valverde et al. (2020)<br>
    PLOS ONE — RF 88.41%<br><br>
    <b style='color:#6b7385'>Model Reviews</b><br>
    Logistic Regression<br>
    Accuracy: 88.28% ± 1.00%<br><br>
    <b style='color:#6b7385'>Model Fraud</b><br>
    Random Forest<br>
    F1: 83.25% · AUC: 98.86%
    </div>
    """, unsafe_allow_html=True)

df_r = load_reviews()
df_f, _fraud_is_dummy = load_fraud()
model_r = load_model_reviews()
model_f = load_model_fraud()

# Tampilkan warning hanya jika benar-benar dummy
if _fraud_is_dummy:
    st.sidebar.warning("⚠️ fraud_cleaned.csv tidak ditemukan — tampil data sample")


# ══════════════════════════════════════════════════════════
# PAGE 1 — RINGKASAN
# ══════════════════════════════════════════════════════════

if "Ringkasan" in page:
    st.markdown("""
    <div class='dash-header'>
        <p class='dash-title'>Dari Persepsi ke Transaksi</p>
        <p class='dash-sub'>Analisis Perilaku Pengguna Layanan Fintech — Reviews · Fraud · Integrasi</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='metric-row'>
        <div class='metric-card'>
            <p class='metric-label'>Total Ulasan</p>
            <p class='metric-value'>15.746</p>
            <p class='metric-sub'>Celsius Network</p>
        </div>
        <div class='metric-card'>
            <p class='metric-label'>Accuracy Model Sentimen</p>
            <p class='metric-value'>88.28%</p>
            <p class='metric-sub'>±1.00% · CV 10-fold</p>
        </div>
        <div class='metric-card'>
            <p class='metric-label'>Total Transaksi</p>
            <p class='metric-value'>1.048.575</p>
            <p class='metric-sub'>PaySim simulation</p>
        </div>
        <div class='metric-card'>
            <p class='metric-label'>AUC Fraud Model</p>
            <p class='metric-value'>98.86%</p>
            <p class='metric-sub'>Random Forest</p>
        </div>
        <div class='metric-card'>
            <p class='metric-label'>Selisih dari Jurnal</p>
            <p class='metric-value'>−0.13%</p>
            <p class='metric-sub'>Carbo-Valverde 2020</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p class='section-title'>Roadmap penelitian</p>", unsafe_allow_html=True)
        stages = [
            ("Tahap 1", "EDA", True),
            ("Tahap 2", "Preprocessing", True),
            ("Tahap 3", "Modelling", True),
            ("Tahap 4", "Evaluasi (CV 10-fold)", True),
            ("Tahap 5", "Integrasi Insight", True),
            ("Tahap 6", "Dashboard ← sekarang", True),
            ("Tahap 7", "Deployment", False),
        ]
        for code, name, done in stages:
            color = "#34d399" if done else "#4e5568"
            icon  = "✓" if done else "○"
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;
                        padding:8px 12px;margin-bottom:4px;
                        background:#1a1f2e;border-radius:8px;
                        border:1px solid {"#0d2e1f" if done else "#1e2130"}'>
                <span style='color:{color};font-size:14px;width:16px'>{icon}</span>
                <span style='color:#6b7385;font-size:11px;width:60px'>{code}</span>
                <span style='color:{"#c9cdd6" if done else "#4e5568"};font-size:13px'>{name}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<p class='section-title'>Perbandingan dengan jurnal</p>", unsafe_allow_html=True)
        fig, ax = dark_fig((6,3.5))
        models = ['Jurnal\n(RF)', 'Kita\n(LR·CV)', 'CI Bawah', 'CI Atas']
        vals   = [88.41, 88.28, 86.27, 90.29]
        colors = [PALETTE['accent'], PALETTE['positive'], PALETTE['muted'], PALETTE['muted']]
        bars = ax.barh(models, vals, color=colors, height=0.5)
        ax.set_xlim(84, 92)
        ax.axvline(88.41, color=PALETTE['accent'], linewidth=1, linestyle='--', alpha=0.5)
        for bar, val in zip(bars, vals):
            ax.text(val+0.05, bar.get_y()+bar.get_height()/2,
                    f'{val:.2f}%', va='center', color='#c9cdd6', fontsize=9)
        ax.set_xlabel('Accuracy (%)', color='#6b7385')
        ax.set_title('Accuracy vs Carbo-Valverde (2020)', color='#c9cdd6', fontsize=11, pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("""
    <div class='insight-box'>
    <strong>Insight utama penelitian:</strong> Sentimen negatif pengguna fintech bukan sekadar keluhan —
    ia adalah sinyal <em>early warning</em> risiko platform. Kata "withdraw" dan "locked" di ulasan
    berkorelasi langsung dengan pola fraud CASH_OUT dan TRANSFER di data transaksi. Krisis Celsius
    Network Juni 2022 terbukti terdeteksi lebih awal melalui tren sentimen (69.93% negatif) sebelum
    tindakan platform diumumkan secara resmi.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 2 — REVIEWS & SENTIMEN
# ══════════════════════════════════════════════════════════

elif "Reviews" in page:
    st.markdown("<p class='dash-title' style='font-family:DM Serif Display,serif;color:#f0f2f8;margin-bottom:4px'>Reviews & Sentimen</p>", unsafe_allow_html=True)
    st.markdown("<p class='dash-sub' style='color:#6b7385;margin-bottom:20px'>Dataset Celsius Network · 15.746 ulasan · Logistic Regression</p>", unsafe_allow_html=True)

    # Metrics
    neg_pct = (df_r['sentiment']=='negative').mean()*100
    pos_pct = (df_r['sentiment']=='positive').mean()*100
    neu_pct = (df_r['sentiment']=='neutral').mean()*100
    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card'><p class='metric-label'>Positif</p>
            <p class='metric-value' style='color:#34d399'>{pos_pct:.1f}%</p>
            <p class='metric-sub'>{int(pos_pct/100*len(df_r)):,} ulasan</p></div>
        <div class='metric-card'><p class='metric-label'>Negatif</p>
            <p class='metric-value' style='color:#f87171'>{neg_pct:.1f}%</p>
            <p class='metric-sub'>{int(neg_pct/100*len(df_r)):,} ulasan</p></div>
        <div class='metric-card'><p class='metric-label'>Netral</p>
            <p class='metric-value' style='color:#fbbf24'>{neu_pct:.1f}%</p>
            <p class='metric-sub'>{int(neu_pct/100*len(df_r)):,} ulasan</p></div>
        <div class='metric-card'><p class='metric-label'>Accuracy (CV)</p>
            <p class='metric-value'>88.28%</p>
            <p class='metric-sub'>±1.00% · 10-fold</p></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p class='section-title'>Distribusi sentimen</p>", unsafe_allow_html=True)
        fig, ax = dark_fig((5,3.5))
        sizes  = [pos_pct, neg_pct, neu_pct]
        labels = ['Positif','Negatif','Netral']
        colors = [PALETTE['positive'], PALETTE['negative'], PALETTE['neutral']]
        wedges, texts, autos = ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, pctdistance=0.78,
            wedgeprops=dict(width=0.55, edgecolor='#0f1117', linewidth=2))
        for t in texts:  t.set_color('#c9cdd6'); t.set_fontsize(10)
        for a in autos:  a.set_color('#0f1117'); a.set_fontsize(9); a.set_fontweight('500')
        ax.text(0,0,f'{len(df_r):,}\nulasan', ha='center', va='center',
                color='#c9cdd6', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<p class='section-title'>Distribusi bintang</p>", unsafe_allow_html=True)
        fig, ax = dark_fig((5,3.5))
        star_counts = df_r['stars'].value_counts().sort_index()
        bar_colors  = [PALETTE['negative'],PALETTE['negative'],PALETTE['neutral'],
                       PALETTE['positive'],PALETTE['positive']]
        bars = ax.bar(star_counts.index, star_counts.values,
                      color=bar_colors[:len(star_counts)], width=0.6)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                    f'{int(bar.get_height()):,}', ha='center', color='#6b7385', fontsize=8)
        ax.set_xlabel('Rating bintang'); ax.set_ylabel('Jumlah ulasan')
        ax.set_title('Rating distribution', color='#c9cdd6', fontsize=11, pad=10)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Tren temporal
    if 'year' in df_r.columns and 'month' in df_r.columns:
        st.markdown("<p class='section-title'>Tren sentimen negatif 2022</p>", unsafe_allow_html=True)
        df_r2 = df_r.dropna(subset=['year','month']).copy()
        df_r2['year']  = df_r2['year'].astype(int)
        df_r2['month'] = df_r2['month'].astype(int)
        df_22 = df_r2[df_r2['year']==2022].copy()
        if len(df_22) > 0:
            trend = df_22.groupby('month').apply(
                lambda x: (x['sentiment']=='negative').mean()*100).reset_index()
            trend.columns = ['month','neg_pct']
            bulan = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mei',6:'Jun',
                     7:'Jul',8:'Agu',9:'Sep',10:'Okt',11:'Nov',12:'Des'}
            trend['label'] = trend['month'].map(bulan)
            fig, ax = dark_fig((10,3))
            ax.fill_between(trend['label'], trend['neg_pct'], alpha=0.15, color=PALETTE['negative'])
            ax.plot(trend['label'], trend['neg_pct'], color=PALETTE['negative'],
                    linewidth=2, marker='o', markersize=5,
                    markerfacecolor='#0f1117', markeredgewidth=2)
            # Annotate peak
            peak_idx = trend['neg_pct'].idxmax()
            ax.annotate(f"Puncak krisis\n{trend.loc[peak_idx,'neg_pct']:.1f}%",
                        xy=(trend.loc[peak_idx,'label'], trend.loc[peak_idx,'neg_pct']),
                        xytext=(peak_idx-1 if peak_idx>1 else peak_idx+1,
                                trend.loc[peak_idx,'neg_pct']+5),
                        fontsize=8, color='#f87171',
                        arrowprops=dict(arrowstyle='->', color='#f87171', lw=1))
            ax.set_ylabel('% ulasan negatif'); ax.set_ylim(0,80)
            ax.set_title('Rasio sentimen negatif per bulan 2022', color='#c9cdd6', fontsize=11, pad=10)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    st.markdown("""
    <div class='insight-box'>
    <strong>Temuan kunci:</strong> Rasio ulasan negatif melonjak ke <strong>69.93%</strong> pada Juni 2022 —
    bertepatan dengan krisis Celsius Network yang membekukan penarikan dana. Ulasan negatif rata-rata
    lebih panjang (262 karakter) dibanding positif (204 karakter), menunjukkan pengguna yang kecewa
    lebih detail dalam mengekspresikan keluhan.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 3 — FRAUD DETECTION
# ══════════════════════════════════════════════════════════

elif "Fraud" in page:
    st.markdown("<p class='dash-title' style='font-family:DM Serif Display,serif;color:#f0f2f8;margin-bottom:4px'>Fraud Detection</p>", unsafe_allow_html=True)
    st.markdown("<p class='dash-sub' style='color:#6b7385;margin-bottom:20px'>Dataset PaySim · 1.048.575 transaksi · Random Forest</p>", unsafe_allow_html=True)

    fraud_col = 'is_fraud' if 'is_fraud' in df_f.columns else 'isFraud'
    n_fraud = int(df_f[fraud_col].sum())
    n_total = len(df_f)
    fraud_pct = n_fraud/n_total*100

    fraud_only  = df_f[df_f[fraud_col]==1]
    normal_only = df_f[df_f[fraud_col]==0]
    avg_fraud  = fraud_only['amount'].mean()
    avg_normal = normal_only['amount'].mean()

    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card'><p class='metric-label'>Total Transaksi</p>
            <p class='metric-value'>{n_total:,}</p></div>
        <div class='metric-card'><p class='metric-label'>Kasus Fraud</p>
            <p class='metric-value' style='color:#f87171'>{n_fraud:,}</p>
            <p class='metric-sub'>{fraud_pct:.4f}% dari total</p></div>
        <div class='metric-card'><p class='metric-label'>F1-Score</p>
            <p class='metric-value'>83.25%</p></div>
        <div class='metric-card'><p class='metric-label'>ROC-AUC</p>
            <p class='metric-value'>98.86%</p></div>
        <div class='metric-card'><p class='metric-label'>Avg Amount Fraud</p>
            <p class='metric-value' style='color:#f87171'>{avg_fraud/avg_normal:.1f}x</p>
            <p class='metric-sub'>lebih besar dari normal</p></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p class='section-title'>Fraud per jenis transaksi</p>", unsafe_allow_html=True)
        tc = df_f.groupby('transaction_type')[fraud_col].sum().sort_values(ascending=False)
        fig, ax = dark_fig((5,3.5))
        colors_tc = [PALETTE['negative'] if v>0 else PALETTE['muted'] for v in tc.values]
        ax.bar(tc.index, tc.values, color=colors_tc, width=0.6)
        for i,(idx,val) in enumerate(tc.items()):
            ax.text(i, val+1, str(int(val)), ha='center', color='#6b7385', fontsize=9)
        ax.set_ylabel('Jumlah fraud'); ax.set_title('Fraud per tipe transaksi', color='#c9cdd6', fontsize=11, pad=10)
        plt.xticks(rotation=15)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<p class='section-title'>Distribusi amount (log scale)</p>", unsafe_allow_html=True)
        fig, ax = dark_fig((5,3.5))
        ax.hist(np.log1p(normal_only['amount'].sample(min(5000,len(normal_only)))),
                bins=40, alpha=0.6, color=PALETTE['legit'], label='Legitimate', density=True)
        ax.hist(np.log1p(fraud_only['amount']),
                bins=40, alpha=0.7, color=PALETTE['fraud'], label='Fraud', density=True)
        ax.legend(facecolor='#1a1f2e', labelcolor='#c9cdd6', fontsize=9)
        ax.set_xlabel('log(Amount + 1)'); ax.set_ylabel('Density')
        ax.set_title('Amount distribution', color='#c9cdd6', fontsize=11, pad=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("""
    <div class='insight-box'>
    <strong>Temuan kunci:</strong> Fraud <strong>100%</strong> hanya terjadi di CASH_OUT (578 kasus)
    dan TRANSFER (564 kasus). Rata-rata nilai transaksi fraud <strong>7.6× lebih besar</strong>
    dari transaksi normal. Flag <code>zero_orig_balance</code> — saldo pengirim nol sebelum transaksi
    — merupakan indikator kuat kecurangan.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 4 — INTEGRASI INSIGHT
# ══════════════════════════════════════════════════════════

elif "Integrasi" in page:
    st.markdown("<p class='dash-title' style='font-family:DM Serif Display,serif;color:#f0f2f8;margin-bottom:4px'>Integrasi Insight</p>", unsafe_allow_html=True)
    st.markdown("<p class='dash-sub' style='color:#6b7385;margin-bottom:20px'>Menghubungkan persepsi pengguna dengan risiko transaksi</p>", unsafe_allow_html=True)

    # Risk keywords chart
    st.markdown("<p class='section-title'>Integrasi 1 — kata risiko di ulasan negatif (per 1.000 ulasan)</p>", unsafe_allow_html=True)
    words  = ['fraud','locked','scam','lost','security','stolen','lie','withdraw','steal','gone']
    freqs  = [63.9, 44.3, 41.9, 38.7, 36.9, 34.7, 22.8, 22.5, 20.7, 20.4]
    fig, ax = dark_fig((10,3.2))
    bar_c = [PALETTE['negative'] if f>40 else
             ('#f59e0b' if f>30 else PALETTE['muted']) for f in freqs]
    bars = ax.barh(words[::-1], freqs[::-1], color=bar_c[::-1], height=0.6)
    for bar,freq in zip(bars, freqs[::-1]):
        ax.text(freq+0.3, bar.get_y()+bar.get_height()/2,
                f'{freq:.1f}x', va='center', color='#c9cdd6', fontsize=9)
    ax.set_xlabel('Frekuensi per 1.000 ulasan negatif')
    ax.set_title('Top 10 kata risiko di ulasan negatif', color='#c9cdd6', fontsize=11, pad=10)
    ax.set_xlim(0,75)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background:#2e0d0d;border-radius:10px;padding:16px;'>
        <p style='font-size:11px;color:#f87171;font-weight:500;letter-spacing:.06em;text-transform:uppercase;margin:0 0 8px'>Temuan 1 — Tematik</p>
        <p style='font-size:13px;color:#fca5a5;margin:0;line-height:1.7'>
        "fraud" muncul 63.9× per 1.000 ulasan negatif. Tema keamanan, penarikan dana, dan pembekuan akun mendominasi keluhan pengguna.
        </p></div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#2e2100;border-radius:10px;padding:16px;'>
        <p style='font-size:11px;color:#fbbf24;font-weight:500;letter-spacing:.06em;text-transform:uppercase;margin:0 0 8px'>Temuan 2 — Temporal</p>
        <p style='font-size:13px;color:#fde68a;margin:0;line-height:1.7'>
        Juni 2022: 69.93% ulasan negatif bertepatan krisis Celsius. Sentimen negatif dapat jadi early warning sebelum platform mengumumkan masalah.
        </p></div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background:#0d1a2e;border-radius:10px;padding:16px;'>
        <p style='font-size:11px;color:#60a5fa;font-weight:500;letter-spacing:.06em;text-transform:uppercase;margin:0 0 8px'>Temuan 3 — Profil</p>
        <p style='font-size:13px;color:#93c5fd;margin:0;line-height:1.7'>
        Fraud amount 7.6× lebih besar dari normal. 100% terjadi di CASH_OUT & TRANSFER — konsisten dengan kata "withdraw" di ulasan.
        </p></div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <strong>Koneksi Persepsi ↔ Risiko:</strong><br>
    • Keluhan keamanan di ulasan → sejalan dengan pola fraud di data transaksi<br>
    • Pengguna yang mengeluh "withdraw / locked" → indikasi pengalaman fraud nyata<br>
    • Tren sentimen negatif temporal → dapat jadi <em>early warning system</em> risiko platform fintech
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 5 — PREDIKSI REAL-TIME
# ══════════════════════════════════════════════════════════

elif "Prediksi" in page:
    st.markdown("<p class='dash-title' style='font-family:DM Serif Display,serif;color:#f0f2f8;margin-bottom:4px'>Prediksi Real-time</p>", unsafe_allow_html=True)
    st.markdown("<p class='dash-sub' style='color:#6b7385;margin-bottom:20px'>Uji model langsung dengan data baru</p>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💬 Prediksi Sentimen", "🔍 Prediksi Fraud"])

    with tab1:
        st.markdown("#### Prediksi sentimen ulasan fintech")
        review_input = st.text_area(
            "Masukkan teks ulasan:",
            placeholder="Contoh: I lost all my money and cannot withdraw. The platform is a scam.",
            height=120
        )
        if st.button("Analisis Sentimen ↗", key="btn_sentiment"):
            if review_input.strip():
                if model_r is not None:
                    cleaned = clean_text(review_input)
                    pred = model_r.predict([cleaned])[0]
                    proba = model_r.predict_proba([cleaned])[0]
                    label_map = {0:'Negatif', 1:'Netral', 2:'Positif'}
                    cls_map   = {0:'pred-negative', 1:'pred-neutral', 2:'pred-positive'}
                    label = label_map.get(pred, str(pred))
                    cls   = cls_map.get(pred, 'pred-neutral')
                    conf  = max(proba)*100
                    st.markdown(f"""
                    <div class='pred-result'>
                        <p class='pred-label'>Hasil prediksi model</p>
                        <p class='pred-value {cls}'>{label}</p>
                        <p style='color:#6b7385;font-size:13px;margin:8px 0 0'>
                            Confidence: {conf:.1f}%
                        </p>
                    </div>""", unsafe_allow_html=True)
                    cols = st.columns(3)
                    lab = ['Negatif','Netral','Positif']
                    col = [PALETTE['negative'],PALETTE['neutral'],PALETTE['positive']]
                    for i,(c,p) in enumerate(zip(cols, proba)):
                        c.metric(lab[i], f"{p*100:.1f}%")
                else:
                    st.info("Model belum dimuat. Pastikan `best_model_reviews.pkl` tersedia.")
            else:
                st.warning("Masukkan teks ulasan terlebih dahulu.")

    with tab2:
        st.markdown("#### Prediksi risiko fraud transaksi")
        col1, col2 = st.columns(2)
        with col1:
            tx_type   = st.selectbox("Jenis transaksi", ['CASH_OUT','TRANSFER','PAYMENT','CASH_IN','DEBIT'])
            amount    = st.number_input("Jumlah transaksi", min_value=0.0, value=500000.0, step=10000.0)
            old_orig  = st.number_input("Saldo asal sebelum", min_value=0.0, value=0.0, step=10000.0)
        with col2:
            new_orig  = st.number_input("Saldo asal sesudah", min_value=0.0, value=0.0, step=10000.0)
            old_dest  = st.number_input("Saldo tujuan sebelum", min_value=0.0, value=100000.0, step=10000.0)
            new_dest  = st.number_input("Saldo tujuan sesudah", min_value=0.0, value=600000.0, step=10000.0)

        if st.button("Analisis Risiko Fraud ↗", key="btn_fraud"):
            zero_bal  = 1 if old_orig == 0 else 0
            bal_ch_o  = new_orig - old_orig
            bal_ch_d  = new_dest - old_dest
            is_merch  = 0
            type_enc  = {'CASH_OUT':0,'DEBIT':1,'PAYMENT':2,'TRANSFER':3,'CASH_IN':4}
            t_num     = type_enc.get(tx_type, 0)

            # Heuristic jika model tidak tersedia
            risk_score = 0.0
            if tx_type in ['CASH_OUT','TRANSFER']:
                risk_score += 0.4
            if zero_bal == 1:
                risk_score += 0.3
            if amount > 1000000:
                risk_score += 0.2
            if bal_ch_o < -amount*0.9:
                risk_score += 0.1
            risk_score = min(risk_score, 0.95)

            if model_f is not None:
                try:
                    X = np.array([[t_num, amount, old_orig, new_orig, old_dest,
                                   new_dest, zero_bal, is_merch, bal_ch_o, bal_ch_d]])
                    pred_f = model_f.predict(X)[0]
                    prob_f = model_f.predict_proba(X)[0]
                    risk_score = prob_f[1] if len(prob_f)>1 else risk_score
                except Exception:
                    pass

            if risk_score >= 0.5:
                label_f, cls_f = "BERISIKO TINGGI", "pred-negative"
                advice = "Transaksi ini memiliki karakteristik yang mirip dengan pola fraud. Verifikasi identitas pengguna."
            elif risk_score >= 0.3:
                label_f, cls_f = "PERLU PERHATIAN", "pred-neutral"
                advice = "Transaksi ini memiliki beberapa indikator risiko. Pantau aktivitas pengguna."
            else:
                label_f, cls_f = "RISIKO RENDAH", "pred-positive"
                advice = "Transaksi ini tidak menunjukkan pola fraud yang signifikan."

            st.markdown(f"""
            <div class='pred-result'>
                <p class='pred-label'>Hasil analisis risiko</p>
                <p class='pred-value {cls_f}'>{label_f}</p>
                <p style='color:#6b7385;font-size:13px;margin:8px 0 4px'>
                    Risk score: {risk_score*100:.1f}%
                </p>
                <p style='color:#9ca3b8;font-size:12px;margin:0'>{advice}</p>
            </div>""", unsafe_allow_html=True)

            flags = []
            if tx_type in ['CASH_OUT','TRANSFER']: flags.append(('red','Tipe transaksi berisiko'))
            if zero_bal: flags.append(('red','Saldo asal = 0'))
            if amount > 1000000: flags.append(('yellow','Jumlah besar (>1jt)'))
            if not flags: flags.append(('green','Tidak ada flag risiko'))
            flag_html = ''.join([
                f"<span class='badge badge-{c}'>{t}</span>" for c,t in flags
            ])
            st.markdown(f"<div style='margin-top:12px'>{flag_html}</div>", unsafe_allow_html=True)

