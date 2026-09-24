# ```python
import os

# Hide unnecessary TensorFlow startup messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input

import config


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Blood Cell Classifier & Segmenter",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GPU CONFIGURATION
# ============================================================

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Load the trained Xception classification model once
    and reuse it for all uploaded images.
    """
    return keras.models.load_model(config.MODEL_PATH)


# ============================================================
# IMAGE READING
# ============================================================

def decode_uploaded_file(uploaded_file):
    """
    Convert a Streamlit uploaded file into a BGR OpenCV image.
    """

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            f"Could not decode image: {uploaded_file.name}"
        )

    return image


# ============================================================
# SEGMENTATION
# ============================================================

def make_mask(rgb_image):
    """
    Create a binary mask of stained cellular structures.

    Steps:
    1. RGB -> CIELAB
    2. Extract LAB 'a' channel
    3. Gaussian blur
    4. Otsu thresholding
    5. Morphological opening
    6. Morphological closing
    """

    # Convert RGB to CIELAB
    lab = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2LAB
    )

    # 'a' channel separates green <-> magenta/purple information
    a_channel = lab[:, :, 1]

    # Reduce noise
    a_channel = cv2.GaussianBlur(
        a_channel,
        (5, 5),
        0
    )

    # Automatic threshold using Otsu
    _, mask = cv2.threshold(
        a_channel,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological operations
    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    # Remove small noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    # Fill small holes
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=3
    )

    return mask


# ============================================================
# MAIN IMAGE PROCESSING
# ============================================================

def process_image(image_bgr, model):
    """
    Run the complete pipeline:

    Image
       ↓
    Classification preprocessing
       ↓
    Xception model
       ↓
    Class probabilities

    At the same time:

    Image
       ↓
    CIELAB
       ↓
    Otsu threshold
       ↓
    Morphological processing
       ↓
    Connected components
       ↓
    Stained regions
    """

    # --------------------------------------------------------
    # 1. CLASSIFICATION
    # --------------------------------------------------------

    classification_image = cv2.resize(
        image_bgr,
        (
            config.INPUT_SIZE,
            config.INPUT_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )

    # Remove unwanted corner region if enabled
    if config.BLANK_CORNER:
        h, w = classification_image.shape[:2]

        classification_image[
            int(h * config.CORNER_TOP):,
            int(w * config.CORNER_LEFT):
        ] = 255

    # BGR -> RGB
    classification_rgb = cv2.cvtColor(
        classification_image,
        cv2.COLOR_BGR2RGB
    )

    # Xception preprocessing
    X = preprocess_input(
        np.array(
            [classification_rgb],
            dtype=np.float32
        )
    )

    # Model prediction
    logits = model.predict(
        X,
        verbose=0
    )[0]

    # Convert logits to probabilities
    probabilities = tf.nn.softmax(
        logits
    ).numpy()

    predictions = dict(
        zip(
            config.CLASSES,
            probabilities
        )
    )

    # --------------------------------------------------------
    # 2. SEGMENTATION
    # --------------------------------------------------------

    segmentation_image = image_bgr.copy()

    if config.BLANK_CORNER:
        h, w = segmentation_image.shape[:2]

        segmentation_image[
            int(h * config.CORNER_TOP):,
            int(w * config.CORNER_LEFT):
        ] = 255

    # Resize for segmentation
    small_image = cv2.resize(
        segmentation_image,
        config.SEG_SIZE[::-1],
        interpolation=cv2.INTER_AREA
    )

    # BGR -> RGB
    small_rgb = cv2.cvtColor(
        small_image,
        cv2.COLOR_BGR2RGB
    )

    # Create binary mask
    mask = make_mask(
        small_rgb
    )

    # --------------------------------------------------------
    # 3. CONNECTED COMPONENTS
    # --------------------------------------------------------

    number_of_labels, labels = cv2.connectedComponents(
        mask
    )

    # Label 0 is the background
    stained_regions = number_of_labels - 1

    # Percentage of image covered by stain
    stained_percentage = float(
        (mask > 0).mean() * 100
    )

    # --------------------------------------------------------
    # 4. CREATE FULL-SIZE MASK
    # --------------------------------------------------------

    full_mask = cv2.resize(
        mask,
        (
            image_bgr.shape[1],
            image_bgr.shape[0]
        ),
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # 5. DRAW CONTOURS
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        full_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    overlay_bgr = image_bgr.copy()

    cv2.drawContours(
        overlay_bgr,
        contours,
        -1,
        (0, 255, 0),
        3
    )

    # --------------------------------------------------------
    # 6. CONVERT FOR STREAMLIT
    # --------------------------------------------------------

    original_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    overlay_rgb = cv2.cvtColor(
        overlay_bgr,
        cv2.COLOR_BGR2RGB
    )

    return {
        "predictions": predictions,
        "regions": stained_regions,
        "stained_pct": stained_percentage,
        "original": original_rgb,
        "mask": full_mask,
        "overlay": overlay_rgb,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.markdown(
        """
        **Pipeline**

        🖼️ Image  
        ↓  
        🧠 Xception Classification  
        ↓  
        🔬 CIELAB Segmentation  
        ↓  
        🎯 Otsu Thresholding  
        ↓  
        🧩 Connected Components
        """
    )

    st.divider()

    st.markdown("### Model")

    st.write(
        f"Input size: `{config.INPUT_SIZE} × {config.INPUT_SIZE}`"
    )

    st.write(
        f"Classes: `{len(config.CLASSES)}`"
    )

    st.divider()

    st.caption(
        "Blood Cell Classification & Segmentation Project"
    )


# ============================================================
# HEADER
# ============================================================

st.title("🔬 Blood Cell Classifier & Segmenter")

st.markdown(
    """
    Upload microscopic blood smear images to perform:

    - 🧠 **Blood cell classification using Xception**
    - 🎯 **Stained-cell segmentation using CIELAB + Otsu**
    - 🧩 **Connected-component analysis**
    - 📊 **Class probability analysis**
    """
)


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = load_model()

except Exception as e:
    st.error("❌ Could not load the trained model.")

    st.exception(e)

    st.stop()


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_files = st.file_uploader(
    "📁 Upload blood cell image(s)",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tiff"
    ],
    accept_multiple_files=True,
)


# ============================================================
# PROCESS UPLOADED FILES
# ============================================================

if not uploaded_files:

    st.info(
        "👆 Upload one or more blood smear images to begin."
    )

    st.markdown("### Supported formats")

    st.write(
        "JPG • JPEG • PNG • BMP • TIFF"
    )

else:

    st.success(
        f"✅ {len(uploaded_files)} image(s) uploaded."
    )

    for uploaded_file in uploaded_files:

        st.divider()

        st.subheader(
            f"🖼️ {uploaded_file.name}"
        )

        try:

            # ------------------------------------------------
            # READ IMAGE
            # ------------------------------------------------

            image_bgr = decode_uploaded_file(
                uploaded_file
            )

            # ------------------------------------------------
            # PROCESS IMAGE
            # ------------------------------------------------

            with st.spinner(
                f"Analyzing {uploaded_file.name}..."
            ):

                results = process_image(
                    image_bgr,
                    model
                )

            # ------------------------------------------------
            # RANK PREDICTIONS
            # ------------------------------------------------

            ranked_predictions = sorted(
                results["predictions"].items(),
                key=lambda x: x[1],
                reverse=True
            )

            predicted_class = ranked_predictions[0][0]

            confidence = ranked_predictions[0][1]

            # ------------------------------------------------
            # WARNINGS
            # ------------------------------------------------

            if (
                results["stained_pct"]
                < config.MIN_STAINED_PCT
            ):

                st.warning(
                    f"""
                    ⚠️ **Low Stain Warning**

                    Only **{results['stained_pct']:.1f}%**
                    of the image is detected as stained.

                    Very few stained regions were found.
                    Please verify that the blood smear is
                    properly focused and stained.
                    """
                )

            elif (
                results["stained_pct"]
                > config.MAX_STAINED_PCT
            ):

                st.warning(
                    f"""
                    ⚠️ **High Stain Warning**

                    **{results['stained_pct']:.1f}%**
                    of the image is detected as stained.

                    This is higher than the expected range.
                    Please verify slide illumination,
                    image quality, and cleanliness.
                    """
                )

            # ------------------------------------------------
            # SUMMARY METRICS
            # ------------------------------------------------

            st.markdown("### 📊 Analysis Results")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:

                st.metric(
                    "Predicted Class",
                    predicted_class
                )

                st.caption(
                    f"Confidence: {confidence * 100:.2f}%"
                )

            with metric2:

                st.metric(
                    "Stained Regions",
                    results["regions"]
                )

                st.caption(
                    "Connected components"
                )

            with metric3:

                st.metric(
                    "Stained Area",
                    f"{results['stained_pct']:.2f}%"
                )

                st.caption(
                    "Percentage of image"
                )

            # ------------------------------------------------
            # VISUALIZATION
            # ------------------------------------------------

            st.markdown("### 🔬 Image Analysis")

            image_column, prediction_column = st.columns(
                [1.3, 1]
            )

            with image_column:

                view_mode = st.radio(
                    "Choose visualization",
                    [
                        "🟢 Contour Overlay",
                        "🖼️ Original Image",
                        "⚫ Segmentation Mask"
                    ],
                    horizontal=True,
                    key=f"view_{uploaded_file.name}"
                )

                if view_mode == "🟢 Contour Overlay":

                    st.image(
                        results["overlay"],
                        caption="Detected stained regions",
                        use_container_width=True
                    )

                elif view_mode == "🖼️ Original Image":

                    st.image(
                        results["original"],
                        caption="Original blood smear image",
                        use_container_width=True
                    )

                else:

                    st.image(
                        results["mask"],
                        caption="Binary segmentation mask",
                        use_container_width=True
                    )

            # ------------------------------------------------
            # PREDICTION PROBABILITIES
            # ------------------------------------------------

            with prediction_column:

                st.markdown(
                    "#### 🧠 Class Probabilities"
                )

                probability_df = pd.DataFrame(
                    ranked_predictions,
                    columns=[
                        "Cell Class",
                        "Probability"
                    ]
                )

                probability_df["Probability"] = (
                    probability_df["Probability"]
                    .astype(float)
                )

                # Bar chart
                chart_df = probability_df.set_index(
                    "Cell Class"
                )

                st.bar_chart(
                    chart_df
                )

                # Table
                display_df = probability_df.copy()

                display_df["Probability"] = (
                    display_df["Probability"] * 100
                ).map(
                    lambda x: f"{x:.2f}%"
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:

            st.error(
                f"❌ Failed to process `{uploaded_file.name}`"
            )

            st.exception(e)
# ```
