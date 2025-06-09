import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
from laser_holography_simulation import run_simulation  # Make sure your simulation file has this function

st.set_page_config(page_title="Laser Holography Simulation", layout="centered")

st.title("🔬 Laser Holography Simulation")
st.markdown("""
Upload a microscope image of a transparent microsphere illuminated with a red laser.  
The simulation will analyze the diffraction pattern to estimate the refractive index.
""")

uploaded_file = st.file_uploader("📤 Upload your image file", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Running simulation..."):
        # Run the simulation (you need to define run_simulation in your script!)
        try:
            result = run_simulation(img_array)
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.stop()

    st.success("✅ Simulation complete!")
    st.subheader("📊 Resulting Radial Profile")

    fig, ax = plt.subplots()
    ax.plot(result["r"], result["intensity"], label="Simulated Profile")
    ax.set_xlabel("Radius (µm)")
    ax.set_ylabel("Intensity")
    ax.legend()
    st.pyplot(fig)
