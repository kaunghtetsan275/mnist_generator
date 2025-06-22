import streamlit as st
import torch
import matplotlib.pyplot as plt
from model import CVAE, one_hot

model = CVAE()
model.load_state_dict(torch.load("cvae_mnist.pth", map_location="cpu"))
model.eval()

st.title("🖋️ Handwritten Digit Generator")
digit = st.selectbox("Pick a digit (0-9)", list(range(10)))

if st.button("Generate"):
    y = one_hot(torch.tensor([digit]*5)).float()
    z = torch.randn(5, 20)
    with torch.no_grad():
        samples = model.decoder(z, y).view(-1, 28, 28)

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axs):
        ax.imshow(samples[i], cmap="gray")
        ax.axis("off")
    st.pyplot(fig)
