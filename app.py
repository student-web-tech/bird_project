
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "bird_model.keras"
LABELS_PATH = PROJECT_DIR / "models" / "class_names.json"

NUM_TRAINING_IMAGES = 90
CONFIDENCE_THRESHOLD = 60.0

# =====================================================
# 2. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Bird Biodiversity Monitor",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 3. FOREST GREEN THEME
# =====================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F4F7F1;
        color: #20352A;
    }

    [data-testid="stSidebar"] {
        background-color: #173F2B;
    }

    [data-testid="stSidebar"] * {
        color: #F4F7F1;
    }

    .main-title {
        font-size: 2.35rem;
        font-weight: 800;
        color: #174B32;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #607466;
        font-size: 1rem;
        margin-bottom: 1.3rem;
    }

    .section-title {
        color: #174B32;
        font-weight: 700;
        font-size: 1.35rem;
        margin-top: 0.6rem;
    }

    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #DCE7DD;
        box-shadow: 0 3px 12px rgba(25, 65, 42, 0.06);
        min-height: 120px;
    }

    .metric-label {
        color: #64786A;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .metric-value {
        color: #174B32;
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: 8px;
    }

    .result-card {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 22px;
        border: 1px solid #DCE7DD;
    }

    .species-name {
        color: #174B32;
        font-size: 1.45rem;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .footer-note {
        color: #64786A;
        font-size: 0.83rem;
    }

    div.stButton > button {
        background-color: #246442;
        color: white;
        border-radius: 9px;
        border: none;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #174B32;
        color: white;
        border: none;
    }

    [data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 10px;
    }

    h1, h2, h3 {
        color: #174B32;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# 4. LOAD MODEL AND SPECIES LABELS
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def load_species():
    with open(LABELS_PATH, "r") as file:
        return json.load(file)


# =====================================================
# 5. SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("# 🐦 BirdWatch AI")
    st.caption("Biodiversity Monitoring Prototype")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "About the Project",
            "Model Information"
        ]
    )

    st.divider()

    st.markdown("### 🌿 Current Prototype")
    st.write("**Recognition model:** MobileNetV2")
    st.write("**Species classes:** 3")
    st.write("**Dataset images:** 90")

    st.divider()

    st.caption(
        "Developed as an AI-based bird recognition "
        "and biodiversity monitoring project."
    )

# =====================================================
# 6. LOAD LABELS
# =====================================================

try:
    class_names = load_species()
except Exception as error:
    st.error(f"Could not load species labels: {error}")
    st.stop()

# =====================================================
# 7. DASHBOARD PAGE
# =====================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">Bird Biodiversity Monitoring</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered bird image recognition using deep learning '
        'and MobileNetV2 transfer learning.'
        '</div>',
        unsafe_allow_html=True
    )

    # Metric cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">TRAINED SPECIES</div>
                <div class="metric-value">{len(class_names)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">DATASET IMAGES</div>
                <div class="metric-value">{NUM_TRAINING_IMAGES}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">DEEP LEARNING MODEL</div>
                <div class="metric-value">MobileNetV2</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    st.markdown(
        '<div class="section-title">🔍 Bird Species Recognition</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a bird photograph to obtain the model's "
        "most likely species prediction."
    )

    uploaded_file = st.file_uploader(
        "Choose a bird image",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG, JPEG, or PNG bird photograph."
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            # Resize to the model's expected input size
            resized_image = image.resize((160, 160))
            image_array = np.asarray(
                resized_image, dtype=np.float32
            )
            image_array = np.expand_dims(image_array, axis=0)

            with st.spinner("Analyzing the bird image..."):
                model = load_model()

                # Model includes MobileNetV2 preprocessing internally.
                predictions = model.predict(
                    image_array, verbose=0
                )[0]

            predicted_index = int(np.argmax(predictions))
            predicted_species = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

            st.divider()

            image_col, result_col = st.columns(
                [1.05, 1],
                gap="large"
            )

            with image_col:
                st.markdown("### 📷 Uploaded Photograph")

                st.image(
                    image,
                    use_container_width=True
                )

                st.caption(
                    "Image supplied by the user for species recognition."
                )

            with result_col:
                st.markdown("### 🧠 AI Prediction")

                st.markdown(
                    '<div class="result-card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="metric-label">MOST LIKELY CLASS</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="species-name">'
                    f'{predicted_species.replace("_", " ")}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.write("")

                st.metric(
                    "Model confidence",
                    f"{confidence:.2f}%"
                )

                st.progress(
                    min(max(float(predictions[predicted_index]), 0.0), 1.0)
                )

                st.markdown("</div>", unsafe_allow_html=True)

                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning(
                        "Low-confidence prediction. The result may "
                        "be unreliable; verify the bird manually."
                    )
                else:
                    st.success(
                        "The model found a relatively high-scoring match "
                        "among its trained classes."
                    )

            # Confidence scores
            st.divider()
            st.markdown("### 📊 Scores for All Trained Species")

            st.caption(
                "These are the model's output scores, not verified "
                "probabilities of correct identification."
            )

            for index, species in enumerate(class_names):
                score = float(predictions[index])

                score_col, bar_col = st.columns([1, 2])

                with score_col:
                    st.write(species.replace("_", " "))

                with bar_col:
                    st.progress(min(max(score, 0.0), 1.0))
                    st.caption(f"{score * 100:.2f}%")

            # Important model limitation
            st.warning(
                "Model limitation: this prototype was trained on only "
                "three bird species. It always selects the highest-scoring "
                "trained class, even for an unfamiliar species. A score "
                "above 60% does not confirm the bird's identity. The "
                "threshold is only a demonstration setting and is not a "
                "validated unknown-species detector."
            )

        except Exception as error:
            st.error(
                f"Unable to process the image. Details: {error}"
            )

    else:
        st.info(
            "Upload a bird image above to start species recognition."
        )

        st.markdown("### 🌱 How the System Works")

        step1, step2, step3 = st.columns(3)

        with step1:
            st.markdown("#### 1. Upload")
            st.write("Provide a bird photograph.")

        with step2:
            st.markdown("#### 2. Analyze")
            st.write(
                "MobileNetV2 extracts image features and produces "
                "scores for the trained species."
            )

        with step3:
            st.markdown("#### 3. View Results")
            st.write(
                "Review the predicted class and its output score."
            )

    st.divider()

    st.markdown(
        '<div class="footer-note">'
        'BirdWatch AI | Initial research prototype | '
        'Geographic and seasonal analysis modules are under development.'
        '</div>',
        unsafe_allow_html=True
    )

# =====================================================
# 8. ABOUT THE PROJECT
# =====================================================

elif page == "About the Project":

    st.markdown(
        '<div class="main-title">About the Project</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The project aims to develop an AI-based bird biodiversity "
        "monitoring system that identifies bird species from images "
        "and eventually analyzes their geographic, habitat, and "
        "seasonal distribution."
    )

    st.markdown("### 🎯 Main Objectives")

    st.write(
        "1. Identify bird species from photographs using deep learning."
    )

    st.write(
        "2. Support biodiversity monitoring through geographic, "
        "habitat, and seasonal observation analysis."
    )

    st.markdown("### 🛠️ Technology Stack")

    st.write("- Python")
    st.write("- TensorFlow and Keras")
    st.write("- MobileNetV2 transfer learning")
    st.write("- Streamlit dashboard")
    st.write("- SQLite observation database")
    st.write("- NumPy and Pandas for data processing")

    st.markdown("### 📌 Current Implementation Status")

    st.info(
        "Completed: initial three-species image classification "
        "prototype, model training, prediction script, and database "
        "table creation. Pending: automated observation storage, "
        "geographic mapping, habitat and seasonal analysis, and "
        "integration of the complete monitoring dashboard."
    )

# =====================================================
# 9. MODEL INFORMATION
# =====================================================

elif page == "Model Information":

    st.markdown(
        '<div class="main-title">Model Information</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🧠 MobileNetV2")

    st.write(
        "MobileNetV2 is a convolutional neural network architecture "
        "designed for efficient image recognition. This prototype "
        "uses transfer learning from pretrained ImageNet weights, "
        "with the base network frozen while a new classification "
        "layer is trained on the selected bird species."
    )

    st.markdown("### 📚 Training Configuration")

    st.write("- Image size: 160 × 160 pixels")
    st.write("- Number of species: 3")
    st.write("- Dataset size: 90 images")
    st.write("- Training epochs: 3")
    st.write("- Output activation: Softmax")
    st.write("- Loss function: Sparse categorical cross-entropy")
    st.write("- Optimizer: Adam")

    st.markdown("### ⚠️ Known Limitations")

    st.write(
        "- The training dataset is small."
    )

    st.write(
        "- The model is limited to its three trained classes."
    )

    st.write(
        "- High output scores do not guarantee correct identification."
    )

    st.write(
        "- Independent testing is needed to estimate generalization."
    )

    st.write(
        "- Geographic, habitat, seasonal, and distribution-change "
        "analyses have not yet been integrated."
    )

    st.info(
        "Future improvements include expanding the dataset, "
        "evaluating with unseen images, and integrating geotagged "
        "observations for biodiversity analysis."
    )
