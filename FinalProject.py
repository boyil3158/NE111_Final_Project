import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Histogram Distribution Fitter", layout="wide")

#Custom colors to add diversity


st.markdown("""
<style>

    /* APP BACKGROUND */
    .stApp {
        background-color: #121212;
        background-image: linear-gradient(to bottom right, #1a1a1a, #0e0e0e);
    }

    /* MAIN HEADER COLOR */
    h1, h2, h3, h4 {
        color: #FF6868 !important;
    }

    /* SIDEBAR BACKGROUND */
    [data-testid="stSidebar"] {
        background-color: #202020 !important;
    }

    /* BUTTON STYLE */
    .stButton>button {
        background-color: #FF6868;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.2em;
    }

    .stButton>button:hover {
        background-color: #ff8787;
        color: white;
    }

    /* SLIDERS */
    .stSlider > div[data-baseweb="slider"] > div {
        background: #121212 !important;
    }

    /* INPUT BOXES */
    .stTextArea textarea, .stTextInput input {
        background-color: #121212 !important;
        color: white !important;
        border: 1px solid #FF6868 !important;
        border-radius: 8px !important;
    }

</style>
""", unsafe_allow_html=True)

#Titles

st.title("📊 Histogram Distribution Fitter — NE111 Project")
st.write("Upload or paste data, fit distributions, compare models, and manually adjust parameters.")


def parse_manual_data(text):
    try:
        cleaned = text.replace(",", " ")
        arr = np.array([float(x) for x in cleaned.split()])
        return arr
    except:
        return None

def compute_rmse(hist_y, pdf_y):
    return np.sqrt(np.mean((hist_y - pdf_y) ** 2))

def compute_aic_bic(loglik, k, n):
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik
    return aic, bic

# Distributions 
dist_names = {
    "Normal (norm)": stats.norm,
    "Gamma (gamma)": stats.gamma,
    "Weibull Min (weibull_min)": stats.weibull_min,
    "Weibull Max (weibull_max)": stats.weibull_max,
    "Exponential (expon)": stats.expon,
    "Lognormal (lognorm)": stats.lognorm,
    "Beta (beta)": stats.beta,
    "Pareto (pareto)": stats.pareto,
    "Chi-square (chi2)": stats.chi2,
    "Uniform (uniform)": stats.uniform,
    "Laplace (laplace)": stats.laplace,
    "Cauchy (cauchy)": stats.cauchy,
}

# Sidebar - Data Entry

st.sidebar.header("Data Input")
input_mode = st.sidebar.radio("Choose data input method:", ["Upload CSV", "Paste Data"])

user_data = None

if input_mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        numeric_cols = df.select_dtypes(include=["float", "int"]).columns
        if len(numeric_cols) == 0:
            st.sidebar.error("No numeric columns found in CSV.")
        else:
            col_choice = st.sidebar.selectbox("Column to use:", numeric_cols)
            user_data = df[col_choice].dropna().values
            st.sidebar.success(f"Loaded {len(user_data)} data points.")
else:
    text_data = st.sidebar.text_area("Paste values (comma, space, or newline separated):")
    if text_data.strip():
        parsed = parse_manual_data(text_data)
        if parsed is None:
            st.sidebar.error("Invalid number format.")
        else:
            user_data = parsed
            st.sidebar.success(f"Loaded {len(user_data)} data points.")

if user_data is None:
    st.warning("Please provide data using the sidebar.")
    st.stop()


# Tabs for App Navigation

tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualize", "⚙️ Auto-Fit", "🎚️ Manual Fit", "📊 Ranking"]) 

# Tab 1 — Visualization

with tab1:
    st.subheader("Histogram of Data")
    bins = st.slider("Number of bins", 5, 100, 30)

    fig, ax = plt.subplots(figsize=(8, 4))
    hist_vals, hist_edges, _ = ax.hist(user_data, bins=bins, density=True, alpha=0.5, color="#FB54E9")
    ax.set_title("Data Histogram")
    st.pyplot(fig)


# Tab 2 — Auto Fitting

with tab2:
    st.subheader("Automatic Distribution Fitting")
    selected_dist_name = st.selectbox("Choose distribution to fit:", list(dist_names.keys()))
    dist = dist_names[selected_dist_name]

    try:
        params = dist.fit(user_data)
        st.success(f"Fitted parameters: {params}")
    except Exception as e:
        st.error(f"Fit failed: {e}")
        params = None

    if params is not None:
        # PDF curve
        x = np.linspace(min(user_data), max(user_data), 400)
        pdf_vals = dist.pdf(x, *params)

        # Histogram
        hist_y, hist_edges = np.histogram(user_data, bins=30, density=True)
        hist_x = (hist_edges[:-1] + hist_edges[1:]) / 2

        rmse = compute_rmse(np.interp(hist_x, x, pdf_vals), hist_y)

        # log-likelihood
        loglik = np.sum(np.log(dist.pdf(user_data, *params)))
        k = len(params)
        n = len(user_data)
        aic, bic = compute_aic_bic(loglik, k, n)

        st.write(f"**RMSE:** {rmse:.6f}")
        st.write(f"**AIC:** {aic:.2f}")
        st.write(f"**BIC:** {bic:.2f}")

        # KS test
        try:
            ks_stat, ks_p = stats.kstest(user_data, selected_dist_name.split()[0].lower(), params)
            st.write(f"**KS Test:** statistic = {ks_stat:.4f}, p = {ks_p:.4f}")
        except:
            st.write("KS test unavailable for this distribution.")

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.hist(user_data, bins=30, density=True, alpha=0.5, color="#FB54E9")
        ax2.plot(x, pdf_vals, linewidth=2)
        ax2.set_title(f"Fit: {selected_dist_name}")
        st.pyplot(fig2)


# Tab 3 — Manual Fitting

with tab3:
    st.subheader("Manual Parameter Adjustment")
    manual_dist_name = st.selectbox("Distribution", list(dist_names.keys()))
    dist = dist_names[manual_dist_name]

    # Generate sliders for parameters
    st.write("Adjust parameters:")
    # Always have loc and scale; shape parameters vary
    shapes = dist.shapes.split(',') if dist.shapes else []

    shape_vals = []
    for s in shapes:
        val = st.slider(f"Shape parameter: {s}", -5.0, 10.0, 1.0)
        shape_vals.append(val)

    loc = st.slider("loc", min(user_data), max(user_data), float(np.mean(user_data)))
    scale = st.slider("scale", 0.1, max(user_data) - min(user_data), float(np.std(user_data)))

    params = (*shape_vals, loc, scale)

    x = np.linspace(min(user_data), max(user_data), 400)
    pdf_vals = dist.pdf(x, *params)

    hist_y, hist_edges = np.histogram(user_data, bins=30, density=True)
    hist_x = (hist_edges[:-1] + hist_edges[1:]) / 2

    rmse = compute_rmse(np.interp(hist_x, x, pdf_vals), hist_y)
    st.write(f"**RMSE:** {rmse:.6f}")

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.hist(user_data, bins=30, density=True, alpha=0.5, color="#FB54E9")
    ax3.plot(x, pdf_vals, linewidth=2)
    ax3.set_title(f"Manual Fit — {manual_dist_name}")
    st.pyplot(fig3)


# Tab 4 — Auto Ranking of All Distributions

with tab4:
    st.subheader("Distribution Ranking (AIC / BIC / RMSE)")
    rows = []

    for name, d in dist_names.items():
        try:
            p = d.fit(user_data)
            x = np.linspace(min(user_data), max(user_data), 400)
            pdf_vals = d.pdf(x, *p)

            hist_y, hist_edges = np.histogram(user_data, bins=30, density=True)
            hist_x = (hist_edges[:-1] + hist_edges[1:]) / 2
            rmse = compute_rmse(np.interp(hist_x, x, pdf_vals), hist_y)

            loglik = np.sum(np.log(d.pdf(user_data, *p)))
            k = len(p)
            n = len(user_data)
            aic, bic = compute_aic_bic(loglik, k, n)

            rows.append([name, rmse, aic, bic])
        except Exception:
            continue

    df_rank = pd.DataFrame(rows, columns=["Distribution", "RMSE", "AIC", "BIC"])
    df_rank = df_rank.sort_values("AIC")

    st.dataframe(df_rank, use_container_width=True)

    # Allow download as CSV file
    st.download_button("Download ranking as CSV", df_rank.to_csv(index=False), "ranking.csv")
