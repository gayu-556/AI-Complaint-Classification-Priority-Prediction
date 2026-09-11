import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="AI Complaint Management System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
COMPLAINTS_FILE = DATA_DIR / "complaints_db.json"

CATEGORY_MODEL_PATH = BASE_DIR / "models" / "category_model.pkl"
CATEGORY_VECTORIZER_PATH = BASE_DIR / "models" / "category_vectorizer.pkl"
PRIORITY_MODEL_PATH = BASE_DIR / "models" / "priority_model.pkl"
PRIORITY_VECTORIZER_PATH = BASE_DIR / "models" / "priority_vectorizer.pkl"


def ensure_storage():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    if not COMPLAINTS_FILE.exists():
        COMPLAINTS_FILE.write_text("[]", encoding="utf-8")


def load_complaints():
    ensure_storage()
    try:
        with COMPLAINTS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_complaints(records):
    ensure_storage()
    with COMPLAINTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)


@st.cache_resource
def load_models():
    category_model = joblib.load(CATEGORY_MODEL_PATH)
    category_vectorizer = joblib.load(CATEGORY_VECTORIZER_PATH)
    priority_model = joblib.load(PRIORITY_MODEL_PATH)
    priority_vectorizer = joblib.load(PRIORITY_VECTORIZER_PATH)
    return category_model, category_vectorizer, priority_model, priority_vectorizer


def predict_complaint(description, category_model, category_vectorizer, priority_model, priority_vectorizer):
    description = (description or "").strip()
    if not description:
        raise ValueError("Complaint text cannot be empty.")

    category_input = category_vectorizer.transform([description])
    priority_input = priority_vectorizer.transform([description])

    category_pred = category_model.predict(category_input)[0]
    priority_pred = priority_model.predict(priority_input)[0]

    category_conf = float(category_model.predict_proba(category_input).max() * 100)
    priority_conf = float(priority_model.predict_proba(priority_input).max() * 100)

    return {
        "category": category_pred,
        "priority": priority_pred,
        "category_confidence": round(category_conf, 2),
        "priority_confidence": round(priority_conf, 2),
    }


def admin_user():
    return os.getenv("COMPLAINT_ADMIN_USERNAME", "admin")


def admin_password_hash():
    password = os.getenv("COMPLAINT_ADMIN_PASSWORD", "Admin@123")
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def is_valid_admin(username, password):
    return username == admin_user() and hashlib.sha256(password.encode("utf-8")).hexdigest() == admin_password_hash()


def format_status_badge(status):
    badge_colors = {
        "New": "#2563eb",
        "In Progress": "#d97706",
        "Resolved": "#16a34a",
        "Escalated": "#dc2626",
        "Closed": "#6b7280",
    }
    color = badge_colors.get(status, "#475569")
    return f"<span style='color:{color}; font-weight:700;'>{status}</span>"


def add_timeline_entry(record, status, note, actor="System"):
    record.setdefault("timeline", [])
    record["timeline"].append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "note": note,
            "actor": actor,
        }
    )


def initialize_session_state():
    defaults = {
        "complaints": load_complaints(),
        "is_admin": False,
        "selected_example": "",
        "current_admin_user": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()

try:
    category_model, category_vectorizer, priority_model, priority_vectorizer = load_models()
except Exception as exc:
    st.error("Unable to load the trained ML models. Please ensure the model files exist in the models folder.")
    st.code(str(exc))
    st.stop()


st.markdown(
    """
    <style>
    :root {
        --bg: #f3f8ff;
        --panel: #ffffff;
        --primary: #0f172a;
        --secondary: #2563eb;
        --success: #16a34a;
        --warning: #d97706;
        --danger: #dc2626;
        --surface: #eef4ff;
        --muted: #64748b;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: radial-gradient(circle at top left, #edf5ff 0%, #f8fbff 35%, #eef4ff 100%);
        color: #0f172a;
    }

    .main {
        padding-top: 1rem;
    }

    .hero-box {
        background: linear-gradient(135deg, #0b1120 0%, #1d4ed8 52%, #3b82f6 100%);
        color: white;
        border-radius: 28px;
        padding: 30px 32px;
        box-shadow: 0 22px 50px rgba(37, 99, 235, 0.18);
        margin-bottom: 22px;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        color: rgba(255,255,255,0.82);
        max-width: 900px;
    }

    .section-title {
        font-size: 1.7rem;
        font-weight: 800;
        margin: 22px 0 12px 0;
        color: #0f172a;
    }

    .feature-card, .result-card, .mini-panel {
        background: rgba(255,255,255,0.94);
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 16px 34px rgba(15, 23, 42, 0.07);
        border-radius: 22px;
        padding: 22px;
    }

    .mini-panel {
        padding: 18px;
        border-radius: 18px;
    }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f4f8ff 100%);
        border-radius: 18px;
        padding: 16px 18px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.04);
        min-height: 120px;
    }

    .pill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(224, 234, 255, 0.9);
        color: #1d4ed8;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .admin-panel {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #dfe8fd;
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.08);
    }

    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120 0%, #111827 100%);
    }

    section[data-testid="stSidebar"] > div {
        color: white;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        padding: 0.78rem 1.2rem;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.2);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.4);
    }

    .stTextInput input, .stTextArea textarea, .stSelectbox select, .stNumberInput input {
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        padding: 0.72rem 0.8rem;
        background: rgba(255,255,255,0.9);
    }

    textarea {
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## 🛡️ AI Complaint System")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📝 Submit Complaint",
            "📊 Dashboard",
            "🔐 Admin Login",
            "🧑‍💼 Admin Console",
        ],
    )

    st.markdown("---")
    st.caption("ML powered complaint classification")
    st.caption("Priority prediction and admin workflow")

    if st.session_state.get("is_admin"):
        st.success("Admin session active")
        if st.button("Logout", use_container_width=True):
            st.session_state.is_admin = False
            st.session_state.current_admin_user = ""
            st.rerun()


def render_home_page():
    st.markdown(
        """
        <div class="hero-box">
            <div class="pill">AI-driven support operations</div>
            <div class="hero-title">Complaint Management, Reimagined</div>
            <div class="hero-subtitle">
                Turn every complaint into a structured, prioritized, and actionable case with a secure, intelligent workflow designed for faster resolution and better service quality.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = {
        "Total Complaints": len(st.session_state.complaints),
        "High Priority": sum(1 for record in st.session_state.complaints if record.get("priority") == "High"),
        "Resolved": sum(1 for record in st.session_state.complaints if record.get("status") == "Resolved"),
        "Admin Access": "Protected",
    }

    cols = st.columns(4)
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.markdown(
                f"<div class='metric-card'><div style='color:#64748b;font-size:0.82rem;letter-spacing:0.03em;'>{label}</div><div style='font-size:2.1rem;font-weight:800;margin-top:0.5rem;'>{value}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>Why teams choose this system</div>", unsafe_allow_html=True)
    feature_cols = st.columns(4)
    feature_data = [
        ("🤖 AI Classification", "Automatically identifies the correct complaint category using ML-driven text analysis."),
        ("🚨 Smart Priority", "Detects urgency so high-risk complaints are addressed before low-priority issues."),
        ("🔒 Secure Access", "Only authorized administrators can enter the management console and update case status."),
        ("📋 Workflow Tracking", "Every complaint is stored, reviewed, assigned, and resolved with clear action history."),
    ]

    for col, (title, description) in zip(feature_cols, feature_data):
        with col:
            st.markdown(
                f"<div class='feature-card'><div class='pill'>{title.split()[0]}</div><h4>{title}</h4><p>{description}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>How the process works</div>", unsafe_allow_html=True)
    workflow_cols = st.columns(5)
    workflow_steps = [
        ("1", "Submit", "Customer submits a complaint with details."),
        ("2", "Analyze", "AI predicts the category and urgency."),
        ("3", "Review", "Admin checks the case and assigns it."),
        ("4", "Resolve", "Action and notes are recorded."),
        ("5", "Track", "Dashboard monitors progress and outcomes."),
    ]

    for col, (step, label, text) in zip(workflow_cols, workflow_steps):
        with col:
            st.markdown(
                f"<div class='mini-panel'><div style='font-size:1.8rem;font-weight:800;color:#2563eb;'>{step}</div><div style='font-weight:700;margin:8px 0 6px 0;'>{label}</div><div style='color:#475569;line-height:1.5;'>{text}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>Key system advantages</div>", unsafe_allow_html=True)
    st.markdown(
        """
        - Faster identification of recurring and urgent issues
        - Better prioritization for support and operations teams
        - Clear complaint ownership and accountability
        - Safer and more professional complaint handling
        - Improved customer satisfaction through quicker resolutions
        """
    )


def render_submit_page():
    st.markdown("<div class='section-title'>Submit a complaint</div>", unsafe_allow_html=True)
    st.write("Write your complaint in the box below. The AI system will predict the complaint category and priority automatically.")

    with st.form("complaint_form", clear_on_submit=True):
        complaint_text = st.text_area(
            "Complaint",
            height=220,
            placeholder="Type your complaint here...",
        )
        submitted = st.form_submit_button("Analyze and Submit", use_container_width=True)

    if submitted:
        if not complaint_text.strip():
            st.warning("Please enter your complaint before submitting.")
            return

        try:
            prediction = predict_complaint(
                complaint_text,
                category_model,
                category_vectorizer,
                priority_model,
                priority_vectorizer,
            )
        except ValueError as exc:
            st.warning(str(exc))
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "id": f"CMP-{len(st.session_state.complaints) + 1:04d}",
            "name": "Anonymous User",
            "email": "anonymous@example.com",
            "complaint": complaint_text.strip(),
            "category": prediction["category"],
            "priority": prediction["priority"],
            "category_confidence": prediction["category_confidence"],
            "priority_confidence": prediction["priority_confidence"],
            "status": "New",
            "assigned_to": "Unassigned",
            "resolution_note": "",
            "created_at": now,
            "updated_at": now,
            "timeline": [
                {
                    "time": now,
                    "status": "New",
                    "note": "Complaint submitted and AI prediction generated.",
                    "actor": "System",
                }
            ],
        }

        st.session_state.complaints.insert(0, record)
        save_complaints(st.session_state.complaints)

        st.success("Complaint submitted successfully. The admin will review it and take action.")

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("AI prediction summary")
        result_cols = st.columns(2)

        with result_cols[0]:
            st.markdown(f"### 📂 {prediction['category']}")
            st.progress(int(prediction['category_confidence']))
            st.caption(f"Category confidence: {prediction['category_confidence']}%")

        with result_cols[1]:
            priority_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(prediction["priority"], "⚪")
            st.markdown(f"### {priority_icon} {prediction['priority']}")
            st.progress(int(prediction['priority_confidence']))
            st.caption(f"Priority confidence: {prediction['priority_confidence']}%")

        st.markdown(
            f"<div style='margin-top:18px; color:#475569; line-height:1.7;'><strong>Complaint summary:</strong> {complaint_text.strip()}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_dashboard():
    st.markdown("<div class='section-title'>Operations dashboard</div>", unsafe_allow_html=True)

    if not st.session_state.complaints:
        st.info("No complaints have been submitted yet. The dashboard will update as soon as new issues arrive.")
        return

    complaints_df = pd.DataFrame(st.session_state.complaints)

    total = len(complaints_df)
    high = (complaints_df["priority"] == "High").sum()
    medium = (complaints_df["priority"] == "Medium").sum()
    low = (complaints_df["priority"] == "Low").sum()
    resolved = (complaints_df["status"] == "Resolved").sum()

    metric_cols = st.columns(5)
    values = [total, high, medium, low, resolved]
    labels = ["Total", "High", "Medium", "Low", "Resolved"]
    for col, label, value in zip(metric_cols, labels, values):
        with col:
            st.metric(label, value)

    st.markdown("---")
    category_counts = complaints_df["category"].value_counts()
    priority_counts = complaints_df["priority"].value_counts()

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Category Distribution")
        st.bar_chart(category_counts)
    with chart_cols[1]:
        st.subheader("Priority Distribution")
        st.bar_chart(priority_counts)

    st.markdown("---")
    st.subheader("Recent Complaint Records")
    display_df = complaints_df[["id", "name", "category", "priority", "status", "created_at"]].copy()
    st.dataframe(display_df, use_container_width=True)


def render_admin_login():
    st.markdown("<div class='section-title'>Secure admin access</div>", unsafe_allow_html=True)
    st.warning("Only authorized administrators can enter the admin console and manage complaint actions.")

    with st.form("admin_login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login = st.form_submit_button("Login to Admin Portal", use_container_width=True)

    if login:
        if is_valid_admin(username, password):
            st.session_state.is_admin = True
            st.session_state.current_admin_user = username
            st.success("Admin login successful.")
            st.rerun()
        else:
            st.error("Invalid admin credentials.")
            st.caption("Default demo credentials: username = admin, password = Admin@123")


def render_admin_console():
    if not st.session_state.get("is_admin"):
        render_admin_login()
        return

    st.markdown("<div class='section-title'>Admin complaint console</div>", unsafe_allow_html=True)
    st.caption(f"Signed in as: {st.session_state.current_admin_user}")

    if not st.session_state.complaints:
        st.info("There are no complaints to manage yet.")
        return

    complaints_df = pd.DataFrame(st.session_state.complaints)
    status_filter = st.selectbox("Filter by status", ["All", "New", "In Progress", "Resolved", "Escalated", "Closed"])
    if status_filter != "All":
        complaints_df = complaints_df[complaints_df["status"] == status_filter]

    if complaints_df.empty:
        st.info("No complaints match this filter.")
        return

    st.dataframe(
        complaints_df[["id", "name", "category", "priority", "status", "assigned_to", "updated_at"]],
        use_container_width=True,
    )

    st.markdown("---")
    selected_id = st.selectbox("Select complaint to manage", complaints_df["id"].tolist())
    selected_record = next((item for item in st.session_state.complaints if item["id"] == selected_id), None)

    if selected_record is None:
        st.warning("Selected complaint could not be found.")
        return

    st.markdown("<div class='admin-panel'>", unsafe_allow_html=True)
    st.subheader(f"{selected_record['id']} - {selected_record['name']}")
    summary_cols = st.columns(2)

    with summary_cols[0]:
        st.markdown(f"<strong>Complaint:</strong> {selected_record['complaint']}", unsafe_allow_html=True)
        st.markdown(f"<strong>Predicted Category:</strong> {selected_record['category']}", unsafe_allow_html=True)
        st.markdown(f"<strong>Predicted Priority:</strong> {selected_record['priority']}", unsafe_allow_html=True)
        st.markdown(f"<strong>Status:</strong> {format_status_badge(selected_record['status'])}", unsafe_allow_html=True)
        st.markdown(f"<strong>Assigned To:</strong> {selected_record['assigned_to']}", unsafe_allow_html=True)

    with summary_cols[1]:
        st.markdown(f"<strong>Created:</strong> {selected_record.get('created_at', 'N/A')}", unsafe_allow_html=True)
        st.markdown(f"<strong>Last Updated:</strong> {selected_record.get('updated_at', 'N/A')}", unsafe_allow_html=True)
        st.markdown(f"<strong>Resolution Notes:</strong> {selected_record.get('resolution_note', 'No notes yet.') or 'No notes yet.'}", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Complaint timeline")
    timeline = selected_record.get("timeline", [])
    if timeline:
        for item in reversed(timeline):
            st.markdown(
                f"<div class='mini-panel'><div style='font-weight:800; color:#0f172a;'>{item.get('status', 'Updated')}</div><div style='color:#64748b; font-size:0.8rem; margin-top:4px;'>{item.get('time', 'N/A')} • {item.get('actor', 'System')}</div><div style='margin-top:10px; color:#334155;'>{item.get('note', 'No update details.')}</div></div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No status history available yet.")

    st.markdown("---")
    with st.form(f"admin_action_{selected_id}"):
        new_status = st.selectbox(
            "Update status",
            ["New", "In Progress", "Resolved", "Escalated", "Closed"],
            index=["New", "In Progress", "Resolved", "Escalated", "Closed"].index(selected_record.get("status", "New")),
        )
        assigned_to = st.text_input("Assign to team member", value=selected_record.get("assigned_to", "Unassigned"))
        resolution_note = st.text_area("Resolution notes", value=selected_record.get("resolution_note", ""), height=140)

        update = st.form_submit_button("Save Admin Action", use_container_width=True)

    if update:
        for complaint in st.session_state.complaints:
            if complaint["id"] == selected_id:
                previous_status = complaint.get("status", "New")
                complaint["status"] = new_status
                complaint["assigned_to"] = assigned_to.strip() or "Unassigned"
                complaint["resolution_note"] = resolution_note.strip()
                complaint["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if not complaint.get("timeline"):
                    complaint["timeline"] = []

                complaint["timeline"].append(
                    {
                        "time": complaint["updated_at"],
                        "status": new_status,
                        "note": resolution_note.strip() or f"Complaint status updated from {previous_status} to {new_status}.",
                        "actor": st.session_state.get("current_admin_user", "Admin"),
                    }
                )
                break

        save_complaints(st.session_state.complaints)
        st.success("Complaint action saved successfully.")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


page_map = {
    "🏠 Home": render_home_page,
    "📝 Submit Complaint": render_submit_page,
    "📊 Dashboard": render_dashboard,
    "🔐 Admin Login": render_admin_login,
    "🧑‍💼 Admin Console": render_admin_console,
}

page_map[page]()