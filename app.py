import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from laser_holography_simulation import RefractiveIndexEstimator

st.title("Holographic Refractive Index Estimator")

uploaded_file = st.file_uploader("Upload experimental image", type=["png", "jpg", "jpeg", "tif"])

if uploaded_file:
    estimator = RefractiveIndexEstimator()

    # Load and display image
    img_np = estimator.load_experimental_image(uploaded_file)
    st.image(img_np, caption="Uploaded Image", use_column_width=True, clamp=True)

    # Compute experimental radial profile
    exp_radial_profile, r_centers = estimator.radial_profile(img_np)
    exp_minima_positions = estimator.find_intensity_minima(exp_radial_profile, r_centers)

    # Estimate refractive index
    with st.spinner("Estimating refractive index..."):
        best_n, best_cost, best_sim_minima, n_values, costs = estimator.optimize_refractive_index(exp_minima_positions)

    st.success(f"Estimated refractive index: {best_n:.5f}")
    st.write(f"Optimization cost: {best_cost:.4e}")

    # Simulated intensity and radial profile
    sim_intensity = estimator.simulate_imaging_system(best_n)
    sim_radial_profile, sim_r_centers = estimator.radial_profile(sim_intensity)

    # Display simulated diffraction pattern
    st.subheader("Simulated Diffraction Pattern")
    st.image(sim_intensity, caption="Simulated Intensity at Sensor", use_column_width=True, clamp=True)

    # Plot overlaid radial profiles
    fig1, ax1 = plt.subplots()
    ax1.plot(r_centers, exp_radial_profile, label="Experimental")
    ax1.plot(sim_r_centers, sim_radial_profile, label="Simulated", linestyle="--")
    ax1.set_xlabel("Radius (µm)")
    ax1.set_ylabel("Intensity")
    ax1.set_title("Radial Intensity Profiles")
    ax1.legend()
    st.pyplot(fig1)

    # Plot optimization cost curve
    fig2, ax2 = plt.subplots()
    ax2.plot(n_values, costs)
    ax2.axvline(best_n, color='r', linestyle='--', label=f"Best n = {best_n:.5f}")
    ax2.set_xlabel("Refractive Index")
    ax2.set_ylabel("Cost")
    ax2.set_title("Optimization Cost vs Refractive Index")
    ax2.legend()
    st.pyplot(fig2)
