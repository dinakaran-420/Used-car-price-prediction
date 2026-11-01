import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="AutoValue | Car Price Predictor", page_icon="🚘", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:linear-gradient(135deg,#0b1220 0%,#101b2d 55%,#0b1220 100%); color:#eef4ff;}
.block-container {padding-top:1.8rem; max-width:1400px;}
.hero {padding:28px 32px;border:1px solid #263b56;border-radius:22px;background:linear-gradient(115deg,rgba(25,47,73,.95),rgba(13,29,48,.9));margin-bottom:22px;}
.hero h1 {font-family:Manrope,sans-serif;font-size:38px;line-height:1.1;margin:0;color:#f4f8ff;font-weight:800;}
.hero p {color:#9fb2ca;margin:10px 0 0;font-size:15px;}
.eyebrow {color:#61d6c6;font-weight:800;letter-spacing:2px;font-size:11px;margin-bottom:10px;}
.metric-card {background:#14243a;border:1px solid #263d58;border-radius:16px;padding:18px 20px;min-height:108px;}
.metric-label {font-size:12px;color:#9db0c8;text-transform:uppercase;letter-spacing:1px;}
.metric-value {font-size:27px;font-weight:800;color:#f3f7ff;margin-top:8px;}
.metric-sub {font-size:11px;color:#6edacb;margin-top:3px;}
.section-title {font-family:Manrope,sans-serif;font-size:21px;font-weight:800;color:#f2f6ff;margin:8px 0 14px;}
.panel {background:#132238;border:1px solid #263d58;border-radius:18px;padding:20px;}
div[data-testid="stForm"] {background:#132238;border:1px solid #263d58;padding:20px;border-radius:18px;}
.stButton>button,div[data-testid="stFormSubmitButton"] button {background:#36c9b5;color:#071722;border:0;border-radius:10px;font-weight:800;padding:10px 18px;}
.stButton>button:hover,div[data-testid="stFormSubmitButton"] button:hover {background:#63dfcf;color:#071722;}
[data-testid="stSidebar"] {background:#0c1727;border-right:1px solid #263b56;}
h1,h2,h3 {color:#f2f6ff;}
p, label {color:#c5d2e2;}
div[data-baseweb="select"]>div, input, textarea {background:#0e1b2c!important;color:#eef4ff!important;}
.stTabs [data-baseweb="tab"] {color:#a9bad0;}
.stTabs [aria-selected="true"] {color:#5ce0ce!important;}
small {color:#91a7c1;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_resource
def train_models(data):
    target = "Selling_Price"
    features = ["Car_Name", "Year", "Present_Price", "Kms_Driven", "Fuel_Type",
                "Seller_Type", "Transmission", "Owner"]
    X = data[features].copy()
    y = data[target].copy()
    cat = ["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"]
    num = ["Year", "Present_Price", "Kms_Driven", "Owner"]
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cat),
        ("numeric", "passthrough", num)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    candidates = {
        "Random Forest": RandomForestRegressor(n_estimators=350, max_depth=None,
                                                min_samples_split=2, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=350, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=180,
                                                        learning_rate=0.05,
                                                        max_depth=2, random_state=42)
    }
    results, fitted = [], {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred)
        })
        fitted[name] = pipe
    scores = pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True)
    best_name = scores.iloc[0]["Model"]
    return fitted, scores, best_name, X_test, y_test, fitted[best_name].predict(X_test)

st.sidebar.markdown("## 🚘 AutoValue")
st.sidebar.caption("Used car valuation studio")
uploaded = st.sidebar.file_uploader("Upload your car dataset (CSV)", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Navigation**")
page = st.sidebar.radio("Go to", ["Overview", "Predict Price", "Model Performance", "Explore Dataset"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Built with Python · Streamlit · Scikit-learn")

try:
    data = load_data(uploaded) if uploaded else pd.read_csv("car_data.csv")
except Exception as e:
    st.error(f"Could not read the dataset: {e}")
    st.stop()

required = ["Car_Name","Year","Selling_Price","Present_Price","Kms_Driven",
            "Fuel_Type","Seller_Type","Transmission","Owner"]
missing = [c for c in required if c not in data.columns]
if missing:
    st.error("Dataset is missing required columns: " + ", ".join(missing))
    st.stop()
data = data[required].dropna().copy()
if len(data) < 20:
    st.error("Please upload a dataset with at least 20 complete records.")
    st.stop()

with st.spinner("Training and comparing machine learning models..."):
    models, scores, best_name, X_test, y_test, best_pred = train_models(data)

st.markdown("""
<div class="hero">
  <div class="eyebrow">SMART AUTOMOTIVE VALUATION</div>
  <h1>Know what your car<br>is really worth.</h1>
  <p>Explore the market, understand the data, and estimate a used car's resale value with machine learning.</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    st.markdown('<div class="section-title">Your dataset at a glance</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    metrics = [
        ("TOTAL LISTINGS", f"{len(data):,}", "Records available"),
        ("AVERAGE RESALE", f"₹{data.Selling_Price.mean():.2f} L", "Average selling price"),
        ("AVERAGE KILOMETRES", f"{data.Kms_Driven.mean():,.0f}", "Distance driven"),
        ("TOP MODEL", str(data.Car_Name.value_counts().index[0]).title(), "Most frequent in dataset")
    ]
    for col,(label,value,sub) in zip([c1,c2,c3,c4],metrics):
        col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',unsafe_allow_html=True)
    st.write("")
    left,right = st.columns([1.2,1])
    with left:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown("#### Selling price distribution")
        fig,ax=plt.subplots(figsize=(8,3.5))
        fig.patch.set_facecolor("#132238"); ax.set_facecolor("#132238")
        ax.hist(data["Selling_Price"],bins=18,edgecolor="#132238")
        ax.set_xlabel("Selling price (₹ Lakhs)",color="#c5d2e2")
        ax.set_ylabel("Number of cars",color="#c5d2e2")
        ax.tick_params(colors="#9fb2ca")
        for s in ax.spines.values(): s.set_color("#344b65")
        st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown("#### Fuel type mix")
        fuel=data["Fuel_Type"].value_counts()
        fig,ax=plt.subplots(figsize=(5,3.5))
        fig.patch.set_facecolor("#132238"); ax.set_facecolor("#132238")
        ax.pie(fuel.values,labels=fuel.index,autopct="%1.0f%%",startangle=90,textprops={"color":"#eaf1fa"})
        st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.markdown('</div>',unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.markdown("#### Best model on the held-out test set")
    st.markdown(f"**{best_name}** · R² score: **{scores.iloc[0]['R²']:.3f}** · MAE: **₹{scores.iloc[0]['MAE']:.2f} lakh**")
    st.caption("Metrics are calculated from a random 80/20 train-test split. Results can vary with a different dataset.")
    st.markdown('</div>',unsafe_allow_html=True)

elif page == "Predict Price":
    st.markdown('<div class="section-title">Estimate your car value</div>',unsafe_allow_html=True)
    st.write("Enter the vehicle details below. The model uses patterns learned from your uploaded dataset.")
    car_names=sorted(data.Car_Name.astype(str).unique())
    fuel_options=sorted(data.Fuel_Type.astype(str).unique())
    seller_options=sorted(data.Seller_Type.astype(str).unique())
    transmission_options=sorted(data.Transmission.astype(str).unique())
    col1,col2=st.columns(2)
    with st.form("prediction_form"):
        with col1:
            car=st.selectbox("Car model",car_names)
            year=st.number_input("Manufacturing year",min_value=int(data.Year.min()),max_value=2035,value=int(data.Year.median()),step=1)
            present=st.number_input("Present showroom price (₹ lakhs)",min_value=0.1,max_value=500.0,value=float(data.Present_Price.median()),step=0.1)
            kms=st.number_input("Kilometres driven",min_value=0,max_value=1000000,value=int(data.Kms_Driven.median()),step=1000)
        with col2:
            fuel=st.selectbox("Fuel type",fuel_options)
            seller=st.selectbox("Seller type",seller_options)
            transmission=st.selectbox("Transmission",transmission_options)
            owner=st.selectbox("Previous owners",sorted(data.Owner.unique().tolist()))
        submitted=st.form_submit_button("✨ Predict resale price",use_container_width=True)
    if submitted:
        row=pd.DataFrame([{"Car_Name":car,"Year":year,"Present_Price":present,"Kms_Driven":kms,
                           "Fuel_Type":fuel,"Seller_Type":seller,"Transmission":transmission,"Owner":owner}])
        estimate=float(models[best_name].predict(row)[0])
        estimate=max(0,estimate)
        st.markdown("---")
        a,b,c=st.columns([1.4,1,1])
        a.markdown(f'<div class="metric-card"><div class="metric-label">ESTIMATED RESALE VALUE</div><div class="metric-value">₹ {estimate:.2f} L</div><div class="metric-sub">Model estimate · {best_name}</div></div>',unsafe_allow_html=True)
        b.markdown(f'<div class="metric-card"><div class="metric-label">PRESENT PRICE</div><div class="metric-value">₹ {present:.2f} L</div><div class="metric-sub">Input value</div></div>',unsafe_allow_html=True)
        c.markdown(f'<div class="metric-card"><div class="metric-label">CAR AGE</div><div class="metric-value">{max(0,2026-year)} yrs</div><div class="metric-sub">Based on 2026</div></div>',unsafe_allow_html=True)
        st.info("This is an indicative machine-learning estimate, not a guaranteed market offer. Actual value depends on condition, location, service history, accident history and demand.")
        result=pd.DataFrame([{"Car_Name":car,"Year":year,"Present_Price_Lakh":present,"Kms_Driven":kms,
                              "Fuel_Type":fuel,"Seller_Type":seller,"Transmission":transmission,
                              "Owner":owner,"Predicted_Selling_Price_Lakh":round(estimate,2)}])
        st.download_button("Download prediction as CSV",result.to_csv(index=False),"car_price_prediction.csv","text/csv")

elif page == "Model Performance":
    st.markdown('<div class="section-title">Model performance</div>',unsafe_allow_html=True)
    st.caption("Three regression algorithms are trained and compared using the same 80/20 split.")
    st.dataframe(scores.style.format({"MAE":"₹{:.3f} L","RMSE":"₹{:.3f} L","R²":"{:.3f}"}),use_container_width=True,hide_index=True)
    st.markdown(f"**Selected model:** {best_name}")
    m1,m2,m3=st.columns(3)
    best=scores.iloc[0]
    for col,(label,value) in zip([m1,m2,m3],[("R² SCORE",f"{best['R²']:.3f}"),("MAE",f"₹{best['MAE']:.2f} L"),("RMSE",f"₹{best['RMSE']:.2f} L")]):
        col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',unsafe_allow_html=True)
    st.write("")
    fig,ax=plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor("#132238"); ax.set_facecolor("#132238")
    ax.scatter(y_test,best_pred,alpha=.75)
    lim=max(float(y_test.max()),float(np.max(best_pred)))
    ax.plot([0,lim],[0,lim],linestyle="--")
    ax.set_xlabel("Actual price (₹ lakhs)",color="#c5d2e2"); ax.set_ylabel("Predicted price (₹ lakhs)",color="#c5d2e2")
    ax.tick_params(colors="#9fb2ca")
    for s in ax.spines.values(): s.set_color("#344b65")
    st.markdown("#### Actual vs predicted prices")
    st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.warning("Your dataset has 301 rows, which is relatively small. Treat the scores as an initial benchmark; more varied and recent listings can improve reliability.")

elif page == "Explore Dataset":
    st.markdown('<div class="section-title">Explore your car dataset</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        fuel_filter=st.multiselect("Filter by fuel type",sorted(data.Fuel_Type.unique()),default=sorted(data.Fuel_Type.unique()))
    with c2:
        transmission_filter=st.multiselect("Filter by transmission",sorted(data.Transmission.unique()),default=sorted(data.Transmission.unique()))
    filtered=data[data.Fuel_Type.isin(fuel_filter)&data.Transmission.isin(transmission_filter)]
    st.caption(f"Showing {len(filtered)} of {len(data)} records")
    st.dataframe(filtered,use_container_width=True,hide_index=True)
    st.download_button("Download filtered dataset",filtered.to_csv(index=False),"filtered_car_data.csv","text/csv")
    st.markdown("#### Dataset summary")
    st.dataframe(data.describe(include="all").transpose(),use_container_width=True)

st.markdown("---")
st.caption("AutoValue Car Price Predictor · Educational project · Estimates depend on dataset quality and model performance.")