import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import cv2
from datetime import datetime
import hashlib
from PIL import Image
import io
import zipfile
import os
import time
from tensorflow.keras.models import load_model
import requests
import os

def download_file(url, filename):
    if not os.path.exists(filename):
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Discipline Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }
    .compliant { color: green; font-weight: bold; }
    .non-compliant { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load models and classes (cached)
# ------------------------------------------------------------
@st.cache_resource
def load_model_and_classes(model_path, classes_path):
    model = load_model(model_path)
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return model, classes

@st.cache_resource
def load_all_models():
    dress_model, dress_classes = load_model_and_classes("dress_model.h5", "dress_classes.txt")
    grooming_model, grooming_classes = load_model_and_classes("grooming_model.h5", "grooming_classes.txt")
    posture_model, posture_classes = load_model_and_classes("posture_model.h5", "posture_classes.txt")
    id_model, id_classes = load_model_and_classes("id_model.h5", "id_classes.txt")
    return {
        "dress": (dress_model, dress_classes),
        "grooming": (grooming_model, grooming_classes),
        "posture": (posture_model, posture_classes),
        "id": (id_model, id_classes)
    }
download_file(
    "https://huggingface.co/Udayakumar2006/discipline-models/resolve/main/dress_model.h5",
    "dress_model.h5"
)

download_file(
    "https://huggingface.co/Udayakumar2006/discipline-models/resolve/main/grooming_model.h5",
    "grooming_model.h5"
)

download_file(
    "https://huggingface.co/Udayakumar2006/discipline-models/resolve/main/posture_model.h5",
    "posture_model.h5"
)

download_file(
    "https://huggingface.co/Udayakumar2006/discipline-models/resolve/main/id_model.h5",
    "id_model.h5"
)
models = load_all_models()

# ------------------------------------------------------------
# Prediction function (same as original but works on BGR image)
# ------------------------------------------------------------
def predict_all(frame_bgr):
    """frame_bgr: numpy array (H,W,3) in BGR order, as from cv2.imread or webcam"""
    img = cv2.resize(frame_bgr, (128, 128))
    img_array = img.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = {}
    dress_model, dress_classes = models["dress"]
    grooming_model, grooming_classes = models["grooming"]
    posture_model, posture_classes = models["posture"]
    id_model, id_classes = models["id"]

    preds["dresscode"] = dress_classes[np.argmax(dress_model.predict(img_array, verbose=0))]
    preds["bear"] = grooming_classes[np.argmax(grooming_model.predict(img_array, verbose=0))]
    preds["posture"] = posture_classes[np.argmax(posture_model.predict(img_array, verbose=0))]
    preds["card"] = id_classes[np.argmax(id_model.predict(img_array, verbose=0))]
    return preds

def predict_from_uploaded_image(uploaded_file):
    """Convert uploaded file to BGR and run prediction."""
    # Read as PIL, convert to RGB numpy, then to BGR
    pil_img = Image.open(uploaded_file).convert("RGB")
    rgb_img = np.array(pil_img)
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    return predict_all(bgr_img)

# ------------------------------------------------------------
# Dashboard class (modified to use uploaded images)
# ------------------------------------------------------------
class DisciplineDashboard:
    def __init__(self):
        if 'detection_df' not in st.session_state:
            st.session_state.detection_df = pd.DataFrame(columns=[
                'timestamp', 'image_hash', 'person_id', 'dresscode',
                'bear', 'posture', 'card', 'status', 'image_bytes'
            ])
        self.df = st.session_state.detection_df

    def process_uploaded_images(self, uploaded_files):
        new_entries = []
        for file in uploaded_files:
            img_bytes = file.getvalue()
            img_hash = hashlib.sha256(img_bytes).hexdigest()
            if img_hash in self.df['image_hash'].values:
                st.info(f"Skipping duplicate: {file.name}")
                continue

            # Get predictions
            preds = predict_from_uploaded_image(file)
            dress = preds["dresscode"]
            beard = preds["bear"]
            posture = preds["posture"]
            card = preds["card"]

            # Compliance (adjust labels if needed)
            is_compliant = (dress == 'formal' and beard == 'no_beard' and
                            posture == 'tuckin' and card == 'wearing')
            status = 'Compliant' if is_compliant else 'Non-Compliant'

            # Simple person ID: use first 8 chars of hash (no face recognition)
            # You could add face recognition later
            person_id = img_hash[:8]

            new_entry = {
                'timestamp': datetime.now(),
                'image_hash': img_hash,
                'person_id': person_id,
                'dresscode': dress,
                'bear': beard,
                'posture': posture,
                'card': card,
                'status': status,
                'image_bytes': img_bytes
            }
            new_entries.append(new_entry)

        if new_entries:
            new_df = pd.DataFrame(new_entries)
            st.session_state.detection_df = pd.concat([self.df, new_df], ignore_index=True)
            self.df = st.session_state.detection_df
            st.success(f"✅ Processed {len(new_entries)} new image(s).")
        else:
            st.info("No new images to process.")

    # All other methods (create_summary_metrics, create_trend_analysis, etc.)
    # are identical to the previous version – only the analysis part changed.
    # I include them below exactly as before (with small adaptions for image_bytes).
    
    def create_summary_metrics(self):
        if self.df.empty:
            st.warning("No data available. Please upload images.")
            return
        st.markdown('<div class="section-header">📊 Overview Metrics</div>', unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Detections", len(self.df))
        with col2:
            st.metric("Unique Persons", self.df['person_id'].nunique())
        with col3:
            formal = (self.df['dresscode'] == 'formal').mean() * 100
            st.metric("Formal Dress", f"{formal:.1f}%")
        with col4:
            clean = (self.df['bear'] == 'no_beard').mean() * 100
            st.metric("Clean Shaven", f"{clean:.1f}%")
        with col5:
            good = (self.df['posture'] == 'tuckin').mean() * 100
            st.metric("Good Posture", f"{good:.1f}%")

    def create_trend_analysis(self):
        if self.df.empty:
            st.warning("No data for trend analysis.")
            return
        st.markdown('<div class="section-header">📈 Detection Trends</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        self.df['date'] = self.df['timestamp'].dt.date
        daily = self.df.groupby('date').size().reset_index(name='count')
        fig1 = px.line(daily, x='date', y='count', title='Daily Detections')
        st.plotly_chart(fig1, use_container_width=True, key="daily")
        self.df['hour'] = self.df['timestamp'].dt.hour
        hourly = self.df['hour'].value_counts().sort_index().reset_index()
        hourly.columns = ['hour', 'count']
        fig2 = px.bar(hourly, x='hour', y='count', title='Detections by Hour')
        st.plotly_chart(fig2, use_container_width=True, key="hourly")

    def create_compliance_analysis(self):
        if self.df.empty:
            st.warning("No data for compliance analysis.")
            return
        st.markdown('<div class="section-header">✅ Compliance Analysis</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        colors = px.colors.qualitative.Plotly
        dress_counts = self.df['dresscode'].value_counts()
        fig1 = px.pie(values=dress_counts.values, names=dress_counts.index, title='Dress Code', color_discrete_sequence=colors)
        st.plotly_chart(fig1, use_container_width=True, key="dress_pie")
        beard_counts = self.df['bear'].value_counts()
        fig2 = px.pie(values=beard_counts.values, names=beard_counts.index, title='Beard', color_discrete_sequence=colors)
        st.plotly_chart(fig2, use_container_width=True, key="beard_pie")
        posture_counts = self.df['posture'].value_counts()
        fig3 = px.pie(values=posture_counts.values, names=posture_counts.index, title='Posture', color_discrete_sequence=colors)
        st.plotly_chart(fig3, use_container_width=True, key="posture_pie")
        card_counts = self.df['card'].value_counts()
        fig4 = px.pie(values=card_counts.values, names=card_counts.index, title='ID Card', color_discrete_sequence=colors)
        st.plotly_chart(fig4, use_container_width=True, key="card_pie")

    def create_person_analysis(self):
        if self.df.empty:
            st.warning("No data for person analysis.")
            return
        st.markdown('<div class="section-header">👥 Person Analysis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        person_counts = self.df['person_id'].value_counts().reset_index()
        person_counts.columns = ['person_id', 'detections']
        fig1 = px.bar(person_counts, x='person_id', y='detections', title='Detections per Person')
        st.plotly_chart(fig1, use_container_width=True, key="person_bar")
        compliance = self.df.groupby('person_id').agg({
            'dresscode': lambda x: (x == 'formal').mean(),
            'bear': lambda x: (x == 'no_beard').mean(),
            'posture': lambda x: (x == 'tuckin').mean(),
            'card': lambda x: (x == 'wearing').mean()
        }).round(2)
        compliance.columns = ['Formal Dress', 'Clean Shaven', 'Good Posture', 'ID Visible']
        fig2 = px.imshow(compliance.T, title='Compliance Heatmap', color_continuous_scale='RdYlGn', aspect='auto')
        st.plotly_chart(fig2, use_container_width=True, key="heatmap")

    def create_image_gallery(self):
        if self.df.empty:
            st.warning("No images to display.")
            return
        st.markdown('<div class="section-header">🖼️ Image Gallery</div>', unsafe_allow_html=True)
        persons = ['All'] + sorted(self.df['person_id'].unique().tolist())
        selected_person = st.selectbox("Filter by Person", persons, key="gallery_person")
        filtered = self.df.copy()
        if selected_person != 'All':
            filtered = filtered[filtered['person_id'] == selected_person]
        if filtered.empty:
            st.info("No images match filters.")
            return
        st.write(f"Showing {len(filtered)} images:")
        for i in range(0, len(filtered), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i+j < len(filtered):
                    row = filtered.iloc[i+j]
                    img = Image.open(io.BytesIO(row['image_bytes']))
                    col.image(img, use_container_width=True)
                    col.markdown(f"""
                    <div class="metric-card">
                        <small>👤 Person: {row['person_id']}</small><br>
                        <small>👔 Dress: {row['dresscode']}</small><br>
                        <small>🧔 Beard: {row['bear']}</small><br>
                        <small>🧍 Posture: {row['posture']}</small><br>
                        <small>🪪 Card: {row['card']}</small><br>
                        <small>⏰ Time: {row['timestamp'].strftime('%H:%M:%S')}</small><br>
                        <small>📊 Status: {row['status']}</small>
                    </div>
                    """, unsafe_allow_html=True)

    def create_real_time_monitor(self):
        if self.df.empty:
            st.warning("No data for real-time monitor.")
            return
        st.markdown('<div class="section-header">🔴 Live Activity Monitor</div>', unsafe_allow_html=True)
        recent = self.df.sort_values('timestamp', ascending=False).head(10)
        for _, row in recent.iterrows():
            compliant = row['status'] == 'Compliant'
            status_color = "🟢 COMPLIANT" if compliant else "🔴 NON-COMPLIANT"
            col1, col2, col3 = st.columns([1,2,1])
            with col1:
                img = Image.open(io.BytesIO(row['image_bytes']))
                st.image(img, width=120)
            with col2:
                st.write(f"**Person ID:** {row['person_id']}")
                st.write(f"**Time:** {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Dress:** {row['dresscode']} | **Beard:** {row['bear']}")
                st.write(f"**Posture:** {row['posture']} | **ID Card:** {row['card']}")
            with col3:
                st.markdown(f"<h3 style='text-align: center;'>{status_color}</h3>", unsafe_allow_html=True)
                if compliant:
                    st.success("All standards met ✅")
                else:
                    st.error("Standards not met ❌")
            st.markdown("---")

    def create_download_section(self):
        if self.df.empty:
            st.warning("No data to download.")
            return
        st.markdown('<div class="section-header">📥 Download Reports & Data</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            csv = self.df.drop(columns=['image_bytes']).to_csv(index=False)
            st.download_button("📊 Download CSV", csv,
                               file_name=f"discipline_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv")
        with col2:
            total = len(self.df)
            unique = self.df['person_id'].nunique()
            dress_rate = (self.df['dresscode'] == 'formal').mean() * 100
            beard_rate = (self.df['bear'] == 'no_beard').mean() * 100
            posture_rate = (self.df['posture'] == 'tuckin').mean() * 100
            card_rate = (self.df['card'] == 'wearing').mean() * 100
            overall = (dress_rate + beard_rate + posture_rate + card_rate) / 4
            report = f"""
AI DISCIPLINE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total detections: {total}
Unique persons: {unique}
Formal dress: {dress_rate:.1f}%
Clean shaven: {beard_rate:.1f}%
Good posture: {posture_rate:.1f}%
ID card worn: {card_rate:.1f}%
Overall compliance: {overall:.1f}%
"""
            st.download_button("📄 Text Report", report, file_name="discipline_report.txt")
        with col3:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zf:
                for idx, row in self.df.iterrows():
                    zf.writestr(f"image_{row['image_hash'][:8]}.jpg", row['image_bytes'])
            st.download_button("🖼️ Download Images (ZIP)", zip_buffer.getvalue(),
                               file_name=f"images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                               mime="application/zip")

# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">🤖 AI Discipline Detection Dashboard</h1>', unsafe_allow_html=True)
    dashboard = DisciplineDashboard()

    with st.sidebar:
        st.markdown("## 📤 Upload Images")
        uploaded_files = st.file_uploader("Choose images...", type=['jpg', 'jpeg', 'png'],
                                          accept_multiple_files=True)
        if uploaded_files:
            if st.button("Process Uploaded Images"):
                with st.spinner("Analyzing images..."):
                    dashboard.process_uploaded_images(uploaded_files)
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        sections = {
            "📊 Overview": "Summary metrics and data preview",
            "📈 Trends": "Detection patterns over time",
            "✅ Compliance": "Standards adherence analysis",
            "👥 Person Analysis": "Individual performance tracking",
            "🖼️ Image Gallery": "Browse captured images",
            "🔴 Real-time Monitor": "Live activity feed",
            "📥 Download Reports": "Export data and reports"
        }
        selected_section = st.radio("Choose a section:", list(sections.keys()),
                                    format_func=lambda x: f"{x} - {sections[x]}")
        st.markdown("---")
        st.markdown("### ℹ️ System Info")
        st.markdown("""
        **Monitors:**
        - 👔 Dress Code (Formal/Informal)
        - 🧔 Facial Hair (Beard/No Beard)
        - 🧍 Posture (Tuckin/Other)
        - 🪪 ID Card (Wearing/Not Wearing)
        """)
        if not dashboard.df.empty:
            st.markdown("---")
            st.markdown("### 📈 Quick Stats")
            st.write(f"**Total Records:** {len(dashboard.df)}")
            st.write(f"**Unique Persons:** {dashboard.df['person_id'].nunique()}")

    # Display selected section
    if selected_section == "📊 Overview":
        dashboard.create_summary_metrics()
        if not dashboard.df.empty:
            st.markdown('<div class="section-header">📋 Recent Detections</div>', unsafe_allow_html=True)
            st.dataframe(dashboard.df[['timestamp', 'person_id', 'dresscode', 'bear', 'posture', 'card', 'status']].head(15))
    elif selected_section == "📈 Trends":
        dashboard.create_trend_analysis()
    elif selected_section == "✅ Compliance":
        dashboard.create_compliance_analysis()
    elif selected_section == "👥 Person Analysis":
        dashboard.create_person_analysis()
    elif selected_section == "🖼️ Image Gallery":
        dashboard.create_image_gallery()
    elif selected_section == "🔴 Real-time Monitor":
        dashboard.create_real_time_monitor()
    elif selected_section == "📥 Download Reports":
        dashboard.create_download_section()

if __name__ == "__main__":
    main()
