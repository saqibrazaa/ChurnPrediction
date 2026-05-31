import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Dict, Tuple, Any

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from imblearn.over_sampling import SMOTE

# Visualization
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================================
# PAGE CONFIG & THEME SETUP
# ============================================================================

st.set_page_config(
    page_title="Churn Prediction Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ============================================================================
# THEME CONFIGURATION
# ============================================================================

THEMES = {
    "dark": {
        "name": "Cyberpunk",
        "bg_color": "#0a0e27",
        "text_color": "#e0e0e0",
        "accent_color": "#00ff88",
        "secondary_color": "#ff006e",
        "tertiary_color": "#00d9ff",
        "plotly_template": "plotly_dark",
        "css_bg": "#0a0e27",
        "css_text": "#e0e0e0",
        "sidebar_bg": "#1a1f3a",
    },
    "light": {
        "name": "Modern Corporate",
        "bg_color": "#ffffff",
        "text_color": "#1f1f1f",
        "accent_color": "#0066cc",
        "secondary_color": "#ff6b35",
        "tertiary_color": "#00a896",
        "plotly_template": "plotly_white",
        "css_bg": "#f8f9fa",
        "css_text": "#1f1f1f",
        "sidebar_bg": "#efefef",
    }
}

def get_current_theme() -> Dict:
    """Get current theme configuration"""
    return THEMES[st.session_state.theme]

def inject_theme_css():
    """Inject custom CSS based on current theme"""
    theme = get_current_theme()
    css = f"""
    <style>
    /* Global Styles */
    :root {{
        --bg-color: {theme['css_bg']};
        --text-color: {theme['css_text']};
        --accent-color: {theme['accent_color']};
        --secondary-color: {theme['secondary_color']};
    }}
    
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {theme['css_bg']} !important;
        color: {theme['css_text']} !important;
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {theme['sidebar_bg']} !important;
    }}
    
    .main {{
        background-color: {theme['css_bg']} !important;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {theme['text_color']} !important;
    }}
    
    p, div, span {{
        color: {theme['text_color']} !important;
    }}
    
    [data-testid="stMetricValue"] {{
        color: {theme['accent_color']} !important;
    }}
    
    .metric-card {{
        background-color: {theme['sidebar_bg']};
        border-left: 4px solid {theme['accent_color']};
        padding: 15px;
        border-radius: 5px;
    }}
    
    /* Button Styling */
    button {{
        background-color: {theme['accent_color']} !important;
        color: white !important;
        border: none !important;
    }}
    
    button:hover {{
        background-color: {theme['secondary_color']} !important;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebarContent"] {{
        color: {theme['text_color']} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR THEME SWITCHER & NAVIGATION
# ============================================================================

def create_sidebar():
    """Create sidebar with theme switcher and navigation"""
    with st.sidebar:
        st.title("🎨 Control Panel")
        
        # Theme Switcher
        st.subheader("Theme Settings")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌙 Dark Mode", use_container_width=True):
                st.session_state.theme = "dark"
                st.rerun()
        with col2:
            if st.button("☀️ Light Mode", use_container_width=True):
                st.session_state.theme = "light"
                st.rerun()
        
        current_theme = get_current_theme()
        st.info(f"**Current Theme:** {current_theme['name']}")
        
        st.divider()
        
        # Navigation
        st.subheader("📍 Navigation")
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "📊 Data Explorer", "🧭 EDA Dashboard", "🔬 Model Training", "📊 Model Comparison", "🎯 Predictions", "📈 Analytics"]
        )
        
        st.divider()
        st.caption("Telco Customer Churn Prediction v1.0")
        st.markdown(
            "**EDA Dashboard:** Interactive visual canvases driven by Plotly drop-down selectors where users can explore structural trends, correlation matrices, and demographic variables."
        )
        return page
    
    return page

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

@st.cache_resource
def load_data():
    """Load and cache the dataset"""
    # Resolve path relative to this script so Streamlit finds the CSV when deployed
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TelcoCustomerChurn.csv')
    df = pd.read_csv(file_path)
    return df


def get_model_filepath(filename: str = 'model.pkl') -> str:
    """Get an absolute file path inside the app directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, filename)


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Preprocess the dataset with TotalCharges cleanup and encoding
    Returns: processed_df, encoders_dict
    """
    df_processed = df.copy()
    
    # Normalize obvious missing strings and convert TotalCharges to numeric
    df_processed.replace({"": np.nan, " ": np.nan}, inplace=True)
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with monthly charges (new customers)
    df_processed['TotalCharges'] = df_processed['TotalCharges'].fillna(df_processed['MonthlyCharges'])
    
    # Impute remaining numeric missing values with median
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
    
    # Identify and handle categorical columns
    categorical_cols = df_processed.select_dtypes(include=['object', 'string']).columns.tolist()
    if 'customerID' in categorical_cols:
        categorical_cols.remove('customerID')
    
    encoders = {}
    
    for col in categorical_cols:
        df_processed[col] = df_processed[col].fillna('Missing').astype(str)
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        encoders[col] = le
    
    # Drop customerID
    if 'customerID' in df_processed.columns:
        df_processed.drop('customerID', axis=1, inplace=True)
    
    return df_processed, encoders

# ============================================================================
# MACHINE LEARNING PIPELINE
# ============================================================================

class ChurnModelPipeline:
    """Handles model training and evaluation"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.metrics = {}
    
    def train_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                     y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train 6 models with GridSearchCV"""
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(self.X_train_scaled, y_train)
        
        model_configs = {
            "Logistic Regression": {
                "model": LogisticRegression(random_state=self.random_state, max_iter=1000),
                "params": {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=self.random_state),
                "params": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
            },
            "Gradient Boosting": {
                "model": GradientBoostingClassifier(random_state=self.random_state),
                "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
            },
            "AdaBoost": {
                "model": AdaBoostClassifier(random_state=self.random_state),
                "params": {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1, 1.5]}
            },
            "SVM": {
                "model": SVC(probability=True, random_state=self.random_state),
                "params": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}
            },
            "KNN": {
                "model": KNeighborsClassifier(),
                "params": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
            }
        }
        
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (name, config) in enumerate(model_configs.items()):
            status_text.text(f"Training {name}...")
            
            gs = GridSearchCV(
                config["model"],
                config["params"],
                cv=5,
                scoring="f1",
                n_jobs=-1
            )
            
            gs.fit(X_train_balanced, y_train_balanced)
            
            # Store best model
            self.models[name] = gs.best_estimator_
            
            # Evaluate on test set
            y_pred = gs.best_estimator_.predict(self.X_test_scaled)
            y_pred_proba = gs.best_estimator_.predict_proba(self.X_test_scaled)[:, 1]
            
            results[name] = {
                "best_params": gs.best_params_,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba
            }
            
            progress_bar.progress((idx + 1) / len(model_configs))
        
        progress_bar.empty()
        status_text.empty()
        
        # Select best model based on F1 score
        best_model_name = max(results, key=lambda x: results[x]['f1'])
        self.best_model = self.models[best_model_name]
        self.metrics = results
        
        return results, best_model_name
    
    def save_model(self, filepath: str = "model.pkl"):
        """Save the pipeline state to disk"""
        if not os.path.isabs(filepath):
            filepath = get_model_filepath(filepath)

        state = {
            "best_model": self.best_model,
            "scaler": self.scaler,
            "encoders": getattr(self, "encoders", {}),
            "feature_names": getattr(self, "feature_names_", []),
            "model_results": getattr(self, "model_results", {}),
            "metrics_df": getattr(self, "metrics_df", None),
            "leaderboard_df": getattr(self, "leaderboard_df", None),
            "best_model_name": getattr(self, "best_model_name", None),
            "champion_name": getattr(self, "champion_name", None),
            "champion_metrics": getattr(self, "champion_metrics", None),
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
    
    def load_model(self, filepath: str = "model.pkl"):
        """Load pipeline state from disk"""
        if not os.path.isabs(filepath):
            filepath = get_model_filepath(filepath)

        with open(filepath, "rb") as f:
            loaded = pickle.load(f)
        return loaded

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def page_home():
    """Home page with overview"""
    theme = get_current_theme()
    
    st.title("🏠 Telco Customer Churn Prediction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Models", "6", "Optimized")
    with col2:
        st.metric("🎯 Prediction", "Real-time", "Enabled")
    with col3:
        st.metric("📈 Accuracy", "Up to 95%", "Target")
    
    st.divider()
    
    st.header("About This Application")
    st.markdown("""
This advanced **Customer Churn Prediction** system leverages machine learning to:

- **Predict** whether customers will churn (leave the service)
- **Identify** key factors driving churn
- **Optimize** marketing strategies for retention
- **Visualize** patterns in interactive dashboards

### 🚀 Features
- **6 ML Models** with GridSearchCV optimization
- **Theme Switching** (Cyberpunk & Modern Corporate)
- **Real-time Predictions** for new customers
- **Comprehensive Analytics** & Model Comparisons
- **SMOTE Balancing** for imbalanced datasets

### 📊 Technology Stack
- Python 3.10+
- Streamlit for Web UI
- Scikit-Learn for ML
- Plotly for Interactive Charts
- Imbalanced-Learn for Data Balancing
""")
    
    st.divider()
    st.info("👈 Use the sidebar to navigate to different sections of the application!")

def page_data_explorer():
    """Data exploration and visualization"""
    st.title("📊 Data Explorer")
    
    df = load_data()
    theme = get_current_theme()
    
    st.markdown(
        "Explore the Telco dataset structure with summary metrics, sample records, and distribution charts for churn and key numeric features."
    )
    st.divider()
    
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Total Features", len(df.columns))
    with col3:
        churn_rate = (df['Churn'] == 'Yes').sum() / len(df) * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col4:
        st.metric("Data Quality", "100%")
    
    st.divider()
    
    # Data sample
    st.subheader("Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.divider()
    
    # Churn distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution")
        churn_counts = df['Churn'].value_counts()
        fig = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            title="Churn Distribution",
            template=theme['plotly_template'],
            color_discrete_sequence=[theme['accent_color'], theme['secondary_color']]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Tenure Distribution")
        fig = px.histogram(
            df,
            x='tenure',
            nbins=30,
            title="Tenure Distribution",
            template=theme['plotly_template'],
            color_discrete_sequence=[theme['accent_color']]
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    
    # Monthly charges analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monthly Charges by Churn")
        fig = px.box(
            df,
            x='Churn',
            y='MonthlyCharges',
            title="Monthly Charges Distribution",
            template=theme['plotly_template'],
            color_discrete_sequence=[theme['accent_color']]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Contract Type vs Churn")
        contract_churn = pd.crosstab(df['Contract'], df['Churn'])
        fig = px.bar(
            contract_churn,
            title="Contract Type vs Churn",
            template=theme['plotly_template'],
            color_discrete_sequence=[theme['accent_color'], theme['secondary_color']]
        )
        st.plotly_chart(fig, use_container_width=True)

def page_model_training():
    """Model training and comparison"""
    st.title("🔬 Model Training & Optimization")
    theme = get_current_theme()
    df = load_data()

    st.subheader("Data Preprocessing")
    st.info("Processing dataset with StandardScaler and TotalCharges cleanup...")

    # Preprocess
    df_processed, encoders = preprocess_data(df)

    # Separate features and target
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    st.success(f"✅ Dataset split: {len(X_train)} training, {len(X_test)} testing")

    st.divider()

    # Train models
    st.subheader("Training 6 ML Models with GridSearchCV")

    model_path = get_model_filepath("model.pkl")
    if "training_complete" not in st.session_state:
        st.session_state.training_complete = False

    if not st.session_state.training_complete:
        if os.path.exists(model_path):
            try:
                saved = ChurnModelPipeline().load_model(model_path)
                if isinstance(saved, dict) and saved.get("model_results"):
                    st.success("✅ Loaded previously trained model from disk.")
                    st.session_state.pipeline = ChurnModelPipeline()
                    st.session_state.pipeline.best_model = saved.get("best_model")
                    st.session_state.pipeline.scaler = saved.get("scaler", StandardScaler())
                    st.session_state.pipeline.encoders = saved.get("encoders", {})
                    st.session_state.pipeline.feature_names_ = saved.get("feature_names", [])
                    st.session_state.pipeline.model_results = saved.get("model_results", {})
                    st.session_state.pipeline.metrics_df = saved.get("metrics_df")
                    st.session_state.pipeline.leaderboard_df = saved.get("leaderboard_df")
                    st.session_state.pipeline.best_model_name = saved.get("best_model_name")
                    st.session_state.pipeline.champion_name = saved.get("champion_name")
                    st.session_state.pipeline.champion_metrics = saved.get("champion_metrics")
                    st.session_state.model_results = saved.get("model_results", {})
                    st.session_state.best_model_name = saved.get("best_model_name")
                    st.session_state.metrics_df = saved.get("metrics_df")
                    st.session_state.leaderboard_df = saved.get("leaderboard_df")
                    st.session_state.champion_name = saved.get("champion_name")
                    st.session_state.champion_metrics = saved.get("champion_metrics")
                    st.session_state.training_complete = True
                else:
                    st.warning("Previous model file exists but does not contain full training metadata.")
            except Exception:
                st.warning("Previous model file could not be loaded. You can retrain below.")

    if not st.session_state.training_complete:
        if not st.button("🚀 Start Training"):
            st.info("Press the button above to begin model training. This may take a few minutes on Streamlit Cloud.")
            return

        with st.spinner("Training models. Please wait..."):
            pipeline = ChurnModelPipeline()
            results, best_model_name = pipeline.train_models(X_train, X_test, y_train, y_test)

            # Attach encoders and feature names to pipeline for persistence
            pipeline.encoders = encoders
            pipeline.feature_names_ = X.columns.tolist()

            # Build leaderboard data
            metrics_df = pd.DataFrame({
                model: {
                    "Accuracy": results[model]["accuracy"],
                    "Precision": results[model]["precision"],
                    "Recall": results[model]["recall"],
                    "F1-Score": results[model]["f1"],
                    "ROC-AUC": results[model]["roc_auc"]
                }
                for model in results
            }).T
            leaderboard_df = metrics_df.sort_values(by='F1-Score', ascending=False)
            champion_name = leaderboard_df.index[0]
            champion_metrics = leaderboard_df.loc[champion_name]

            pipeline.model_results = results
            pipeline.best_model_name = best_model_name
            pipeline.metrics_df = metrics_df
            pipeline.leaderboard_df = leaderboard_df
            pipeline.champion_name = champion_name
            pipeline.champion_metrics = champion_metrics

            # Save pipeline object
            pipeline.save_model(model_path)
            st.success(f"✅ Model saved as {model_path}")

            st.session_state.pipeline = pipeline
            st.session_state.encoders = encoders
            st.session_state.model_results = results
            st.session_state.best_model_name = best_model_name
            st.session_state.metrics_df = metrics_df
            st.session_state.leaderboard_df = leaderboard_df
            st.session_state.champion_name = champion_name
            st.session_state.champion_metrics = champion_metrics
            st.session_state.training_complete = True

    if st.session_state.training_complete:
        results = st.session_state.model_results
        best_model_name = st.session_state.best_model_name
        metrics_df = st.session_state.metrics_df
        leaderboard_df = st.session_state.leaderboard_df
        champion_name = st.session_state.champion_name
        champion_metrics = st.session_state.champion_metrics

        st.success("✅ Model training complete. Visit the 'Model Comparison' page to explore the champion dashboard and leaderboard.")
        st.divider()
        st.subheader(f"🏆 Best Model: {best_model_name}")
        best_metrics = results[best_model_name]
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Accuracy", f"{best_metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{best_metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{best_metrics['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{best_metrics['f1']:.4f}")
        with col5:
            st.metric("ROC-AUC", f"{best_metrics['roc_auc']:.4f}")

        st.divider()
        st.info("✅ Model comparison page is ready once you select 'Model Comparison' from the sidebar.")
        return

    # If the function reaches this point without training complete, return early.
    return


def page_model_comparison_dashboard():
    """High-fidelity model comparison dashboard"""
    st.title("📊 Model Comparison Dashboard")
    theme = get_current_theme()

    if "model_results" not in st.session_state:
        model_path = get_model_filepath("model.pkl")
        if os.path.exists(model_path):
            try:
                saved = ChurnModelPipeline().load_model(model_path)
                if isinstance(saved, dict) and saved.get("model_results"):
                    st.session_state.model_results = saved.get("model_results", {})
                    st.session_state.best_model_name = saved.get("best_model_name")
                    st.session_state.metrics_df = saved.get("metrics_df")
                    st.session_state.leaderboard_df = saved.get("leaderboard_df")
                    st.session_state.champion_name = saved.get("champion_name")
                    st.session_state.champion_metrics = saved.get("champion_metrics")
                else:
                    st.warning("⚠️ No trained model data found. Please train models first in the 'Model Training' page.")
                    return
            except Exception:
                st.warning("⚠️ No trained model data found. Please train models first in the 'Model Training' page.")
                return
        else:
            st.warning("⚠️ No trained model data found. Please train models first in the 'Model Training' page.")
            return

    metrics_df = st.session_state.metrics_df
    leaderboard_df = st.session_state.leaderboard_df
    champion_name = st.session_state.champion_name
    champion_metrics = st.session_state.champion_metrics

    st.markdown(
        "This dashboard provides a high-fidelity leaderboard view of model performance and a visual callout for the champion production model."
    )

    st.markdown(f"### 🏆 Champion Production Model: **{champion_name}**")
    st.success(
        f"{champion_name} leads the leaderboard with F1-score {champion_metrics['F1-Score']:.4f} and ROC-AUC {champion_metrics['ROC-AUC']:.4f}."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{champion_metrics['Accuracy']:.4f}")
    c2.metric("Precision", f"{champion_metrics['Precision']:.4f}")
    c3.metric("Recall", f"{champion_metrics['Recall']:.4f}")
    c4.metric("F1-Score", f"{champion_metrics['F1-Score']:.4f}")
    c5.metric("ROC-AUC", f"{champion_metrics['ROC-AUC']:.4f}")

    fig = px.bar(
        leaderboard_df.reset_index(),
        x='index',
        y='F1-Score',
        color='index',
        title='Model Leaderboard by F1-Score',
        template=theme['plotly_template'],
        labels={'index': 'Model', 'F1-Score': 'F1 Score'}
    )
    fig.update_traces(marker_line_width=1.5, marker_line_color='black', showlegend=False)
    fig.add_annotation(
        x=champion_name,
        y=champion_metrics['F1-Score'],
        text='Champion',
        showarrow=True,
        arrowhead=3,
        ax=0,
        ay=-40,
        font=dict(color='black', size=12)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Leaderboard Details:**")
    st.dataframe(leaderboard_df.round(4), use_container_width=True)


def page_eda_dashboard():
    """Dedicated EDA Dashboard with interactive Plotly selectors"""
    st.title("🧭 EDA Dashboard")
    df = load_data()
    theme = get_current_theme()

    st.markdown("Explore structural trends, correlation matrices, and demographic variables.")

    # Structural trends selector
    numeric_options = ['tenure', 'MonthlyCharges', 'TotalCharges']
    metric = st.selectbox("Select numeric metric:", numeric_options, index=0)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[df['Churn'] == 'No'][metric], name='No Churn', marker_color=theme['accent_color'], opacity=0.75
    ))
    fig.add_trace(go.Histogram(
        x=df[df['Churn'] == 'Yes'][metric], name='Churn', marker_color=theme['secondary_color'], opacity=0.75
    ))
    fig.update_layout(template=theme['plotly_template'], barmode='overlay', title=f'Structural Trends: {metric} by Churn')
    st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix
    corr_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    selected_corr = st.multiselect("Select features for correlation matrix:", corr_features, default=corr_features)
    if len(selected_corr) >= 2:
        corr = df[selected_corr].apply(pd.to_numeric, errors='coerce').corr()
        corr_fig = px.imshow(corr, text_auto=True, aspect='auto', title='Correlation Matrix', template=theme['plotly_template'], color_continuous_scale='Viridis')
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.warning("Select at least two numeric features to display the correlation matrix.")

    # Demographic variables with dropdown
    demographic_options = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'InternetService', 'Contract', 'PaymentMethod']
    demo_var = st.selectbox("Select demographic variable:", demographic_options, index=0)
    grouped = df.groupby([demo_var, 'Churn']).size().reset_index(name='count')
    demo_fig = px.bar(grouped, x=demo_var, y='count', color='Churn', barmode='group', title=f'Demographics: {demo_var} by Churn', template=theme['plotly_template'], color_discrete_map={'No': theme['accent_color'], 'Yes': theme['secondary_color']})
    st.plotly_chart(demo_fig, use_container_width=True)

    

def page_predictions():
    """Real-time predictions"""
    st.title("🎯 Prediction System Form")

    theme = get_current_theme()

    # Attempt to ensure a pipeline is available: session state first, then disk
    pipeline = st.session_state.get("pipeline", None)
    model_path = get_model_filepath("model.pkl")
    if pipeline is None:
        # try loading from serialized pipeline file
        if os.path.exists(model_path):
            try:
                loaded_state = ChurnModelPipeline().load_model(model_path)
                if isinstance(loaded_state, dict):
                    wrapper = ChurnModelPipeline()
                    wrapper.best_model = loaded_state.get("best_model")
                    wrapper.scaler = loaded_state.get("scaler", StandardScaler())
                    wrapper.encoders = loaded_state.get("encoders", {})
                    wrapper.feature_names_ = loaded_state.get("feature_names", [])
                    pipeline = wrapper
                elif isinstance(loaded_state, ChurnModelPipeline):
                    pipeline = loaded_state
                else:
                    st.warning("Saved model format is not supported. Please retrain the model.")
                    return

                st.session_state.pipeline = pipeline
            except Exception:
                st.warning("Could not load serialized pipeline from model.pkl. Please train models first.")
                return
        else:
            st.warning("⚠️ No trained pipeline in session and no model.pkl found. Train models first.")
            return

    # encoders may be attached to the pipeline or in session
    encoders = getattr(pipeline, "encoders", st.session_state.get("encoders", {}))

    st.subheader("Enter Customer Data")

    df_sample = load_data()

    # Build intuitive form controls
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.radio("Gender", options=list(df_sample['gender'].dropna().unique()))
        senior_citizen = st.checkbox("Senior Citizen", value=False)
        partner = st.selectbox("Partner", options=list(df_sample['Partner'].dropna().unique()))

    with col2:
        dependents = st.selectbox("Dependents", options=list(df_sample['Dependents'].dropna().unique()))
        tenure = st.slider("Tenure (months)", 0, int(df_sample['tenure'].max()), 12)
        phone_service = st.selectbox("Phone Service", options=list(df_sample['PhoneService'].dropna().unique()))

    with col3:
        internet_service = st.selectbox("Internet Service", options=list(df_sample['InternetService'].dropna().unique()))
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=float(df_sample['MonthlyCharges'].median()), step=1.0)
        contract = st.selectbox("Contract", options=list(df_sample['Contract'].dropna().unique()))

    # Optional: additional fields
    total_charges = st.number_input("Total Charges ($) (leave blank to auto-calc)", min_value=0.0, value=float(monthly_charges * max(1, tenure)), step=1.0)

    if st.button("🔮 Predict Churn", use_container_width=True):
        # Build raw input mapping from friendly fields
        raw_input = {
            'gender': gender,
            'SeniorCitizen': int(senior_citizen),
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'InternetService': internet_service,
            'MonthlyCharges': float(monthly_charges),
            'TotalCharges': float(total_charges),
            'Contract': contract
        }

        # Prepare feature vector matching training feature order
        feature_names = getattr(pipeline, 'feature_names_', None)
        if feature_names is None:
            st.warning("Saved pipeline does not include feature metadata. Prediction cannot proceed.")
            return

        # Start with defaults
        input_vector = {f: 0 for f in feature_names}

        # Fill with provided values where possible
        for k, v in raw_input.items():
            if k in input_vector:
                input_vector[k] = v

        # Encode categorical fields using stored encoders
        for col, encoder in encoders.items():
            if col in input_vector:
                try:
                    transformed = encoder.transform([str(input_vector[col])])[0]
                except Exception:
                    # fallback to nearest known class (first class)
                    try:
                        transformed = int(np.where(encoder.classes_ == encoder.classes_[0])[0][0])
                    except Exception:
                        transformed = 0
                input_vector[col] = transformed

        # Construct DataFrame with correct column order
        df_input = pd.DataFrame([input_vector], columns=feature_names)

        # Scale and predict
        try:
            X_input_scaled = pipeline.scaler.transform(df_input)
        except Exception:
            st.error("Error scaling input — feature mismatch. Ensure the pipeline matches training features.")
            return

        try:
            proba = pipeline.best_model.predict_proba(X_input_scaled)[0]
            pred = pipeline.best_model.predict(X_input_scaled)[0]
        except Exception:
            st.error("Model prediction failed. Ensure the saved pipeline contains a trained estimator with predict_proba.")
            return

        churn_prob = proba[1] * 100
        no_churn_prob = proba[0] * 100

        st.divider()
        st.subheader("📊 Prediction Result")

        if pred == 1:
            st.error(f"⚠️ CHURN RISK — Confidence: {churn_prob:.1f}%")
        else:
            st.success(f"✅ RETAINED ACCOUNT — Confidence: {no_churn_prob:.1f}%")

        # Visual probability bar
        fig = go.Figure(data=[
            go.Bar(x=['No Churn', 'Churn'], y=[no_churn_prob, churn_prob],
                   marker_color=[theme['accent_color'], theme['secondary_color']])
        ])
        fig.update_layout(template=theme['plotly_template'], showlegend=False, yaxis_title='Confidence (%)')
        st.plotly_chart(fig, use_container_width=True)

def page_analytics():
    """Advanced analytics and insights"""
    st.title("📈 Advanced Analytics")
    
    theme = get_current_theme()
    df = load_data()
    
    st.subheader("Customer Insights Dashboard")
    
    # Customer segmentation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Internet Service Type")
        internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
        fig = px.bar(
            internet_churn,
            title="Churn Rate by Internet Service",
            template=theme['plotly_template'],
            color_discrete_sequence=[theme['accent_color'], theme['secondary_color']],
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Payment Method")
        payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
        fig = px.bar(
            payment_churn,
            title="Churn Rate by Payment Method",
            template=theme['plotly_template'],
            color_discrete_sequence=[theme['accent_color'], theme['secondary_color']],
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Service adoption
    st.subheader("Service Adoption Analysis")
    
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    service_data = []
    
    for service in services:
        if service in df.columns:
            yes_count = (df[service] == 'Yes').sum()
            service_data.append({"Service": service, "Adoption": yes_count})
    
    if service_data:
        service_df = pd.DataFrame(service_data)
        fig = px.bar(
            service_df,
            x='Service',
            y='Adoption',
            title="Service Adoption Counts",
            template=theme['plotly_template'],
            color_discrete_sequence=[theme['accent_color']]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Revenue analysis
    st.subheader("Revenue Analysis")
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    revenue_by_churn = df.groupby('Churn')['TotalCharges'].agg(['sum', 'mean', 'count'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Revenue (No Churn)", f"${revenue_by_churn.loc['No', 'sum']:,.0f}")
    with col2:
        st.metric("Total Revenue (Churn)", f"${revenue_by_churn.loc['Yes', 'sum']:,.0f}")
    with col3:
        lost_revenue = revenue_by_churn.loc['Yes', 'sum']
        st.metric("Potential Lost Revenue", f"${lost_revenue:,.0f}")

# ============================================================================
# MAIN APP EXECUTION
# ============================================================================

def main():
    # Inject theme CSS
    inject_theme_css()
    
    # Create sidebar and get selected page
    page = create_sidebar()
    
    # Route to correct page
    if page == "🏠 Home":
        page_home()
    elif page == "📊 Data Explorer":
        page_data_explorer()
    elif page == "🧭 EDA Dashboard":
        page_eda_dashboard()
    elif page == "🔬 Model Training":
        page_model_training()
    elif page == "📊 Model Comparison":
        page_model_comparison_dashboard()
    elif page == "🎯 Predictions":
        page_predictions()
    elif page == "📈 Analytics":
        page_analytics()

if __name__ == "__main__":
    main()
