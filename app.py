"""Streamlit interface for fine-tuned medical-image retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18

from fine_tune_retrieval import make_transforms
from torchvision.models import ResNet18_Weights


ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURES = ROOT / "outputs" / "paper_split" / "resnet18_features.npz"
DEFAULT_CHECKPOINT = ROOT / "outputs" / "paper_split" / "resnet18_finetuned.pt"


@st.cache_resource
def load_retrieval_model(checkpoint_path: str) -> tuple[nn.Module, object, list[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model"])
    model = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    _, evaluation_transform = make_transforms(ResNet18_Weights.IMAGENET1K_V1)
    return model, evaluation_transform, checkpoint["classes"]


def embed_query(model: nn.Module, transform: object, image: Image.Image) -> np.ndarray:
    device = next(model.parameters()).device
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        embedding = model(tensor).flatten(1).cpu().numpy().astype(np.float32)
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    return embedding[0]


def main() -> None:
    st.set_page_config(page_title="Medical Image Retrieval", layout="wide")
    st.title("Medical Image Retrieval")
    st.write("Upload an endoscopy image to retrieve visually similar database images.")

    with st.sidebar:
        st.header("Retrieval settings")
        features_path = st.text_input("Feature file", str(DEFAULT_FEATURES))
        checkpoint_path = st.text_input("Model checkpoint", str(DEFAULT_CHECKPOINT))
        top_k = st.slider("Number of results", min_value=10, max_value=30, value=10)

    feature_file = Path(features_path)
    checkpoint_file = Path(checkpoint_path)
    if not feature_file.exists() or not checkpoint_file.exists():
        st.error("Required model files are missing. Run the fine-tuning command in README.md first.")
        st.stop()

    uploaded_file = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is None:
        st.info("Choose an image to start retrieval.")
        st.stop()

    data = np.load(feature_file, allow_pickle=False)
    model, transform, classes = load_retrieval_model(str(checkpoint_file))
    query_image = Image.open(uploaded_file).convert("RGB")
    query_embedding = embed_query(model, transform, query_image)
    database_indices = np.flatnonzero(data["splits"] == "database")
    database_features = data["features"][database_indices]
    database_features /= np.maximum(np.linalg.norm(database_features, axis=1, keepdims=True), 1e-12)
    scores = database_features @ query_embedding
    order = np.argsort(-scores, kind="stable")[:top_k]

    st.image(query_image, caption="Query image", width=300)
    st.subheader(f"Top {top_k} retrieved images")
    columns = st.columns(5)
    for rank, database_position in enumerate(order, start=1):
        database_index = database_indices[database_position]
        path = Path(str(data["paths"][database_index]))
        with columns[(rank - 1) % len(columns)]:
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.warning(f"Image not found: {path}")
            class_id = int(data["labels"][database_index])
            st.caption(
                f"#{rank} | {str(classes[class_id])} | "
                f"similarity={scores[database_position]:.4f}"
            )


if __name__ == "__main__":
    main()
