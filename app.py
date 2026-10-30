import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(
    page_title="CapsiLeaf: Chili Cultivar Classifier",
    page_icon="🌶️",
    layout="centered"
)

st.title(
    "CapsiLeaf: Mango & Chili Variety Identification")  # Adjust title if you want to keep the exact phrasing from your UI
st.markdown("""
Identify the chili cultivar from its leaf using our ResNet50 model trained on the self-collected CapsiLeaf dataset.
""")

# ==========================================
# 2. MODEL CONFIGURATION
# ==========================================
# The exact 6 classes from your dataset
CLASS_NAMES = [
    "12masi morich (black)",
    "dhani morich",
    "kajoli morich",
    "naga bombai morich",
    "tal morich",
    "thigh chilli pepper"
]

WEIGHTS_PATH = "conference_capsileaf_6_class_lr001.pth"  # Make sure this matches your uploaded weights file name perfectly!


@st.cache_resource
def load_model():
    """Loads the ResNet50 model with the custom 6-class head."""
    try:
        # Initialize ResNet50
        model = models.resnet50(weights=None)

        # Replace the fully connected layer for 6 classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 6)

        # Load weights (map_location='cpu' is critical for Streamlit Cloud)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))

        # Set to evaluation mode
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model weights. Please ensure '{WEIGHTS_PATH}' is in the repository.")
        st.stop()


model = load_model()

# ==========================================
# 3. PREPROCESSING PIPELINE
# ==========================================
# Exact preprocessing pipeline matching your training phase
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# 4. MAIN INFERENCE LOGIC
# ==========================================
uploaded_file = st.file_uploader("Upload a Chili Leaf Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    # Add a submit button to match your UI description in the paper
    if st.button("Submit"):
        with st.spinner("Analyzing leaf venation and morphology..."):
            # Preprocess the image
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch of size 1

            # Forward pass
            with torch.no_grad():
                output = model(input_batch)

            # Apply Softmax to get probabilities
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

            # Sort predictions by confidence
            top_prob, top_catid = torch.topk(probabilities, len(CLASS_NAMES))

            # Display results
            st.success("Analysis Complete!")
            st.subheader(f"Primary Prediction: **{CLASS_NAMES[top_catid[0]].title()}**")

            st.write("### Confidence Scores")
            for i in range(len(CLASS_NAMES)):
                class_name = CLASS_NAMES[top_catid[i]].title()
                score = top_prob[i].item()

                # Create columns for text and progress bar
                col1, col2, col3 = st.columns([3, 5, 2])
                with col1:
                    st.write(f"{class_name}")
                with col2:
                    st.progress(score)
                with col3:
                    st.write(f"{score * 100:.2f}%")

st.markdown("---")
st.caption("Powered by PyTorch & Streamlit | CapsiLeaf IDAA 2027 Research")