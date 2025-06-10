# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# from laser_holography_simulation import RefractiveIndexEstimator

# st.title("Holographic Refractive Index Estimator")

# uploaded_file = st.file_uploader("Upload experimental image", type=["png", "jpg", "jpeg", "tif"])

# if uploaded_file:
#     estimator = RefractiveIndexEstimator()

#     # Load and display image
#     img_np = estimator.load_experimental_image(uploaded_file)
#     st.image(img_np, caption="Uploaded Image", use_column_width=True, clamp=True)

#     # Compute experimental radial profile
#     exp_radial_profile, r_centers = estimator.radial_profile(img_np)
#     exp_minima_positions = estimator.find_intensity_minima(exp_radial_profile, r_centers)

#     # Estimate refractive index
#     with st.spinner("Estimating refractive index..."):
#         best_n, best_cost, best_sim_minima, n_values, costs = estimator.optimize_refractive_index(exp_minima_positions)

#     st.success(f"Estimated refractive index: {best_n:.5f}")
#     st.write(f"Optimization cost: {best_cost:.4e}")

#     # Simulated intensity and radial profile
#     sim_intensity = estimator.simulate_imaging_system(best_n)
#     sim_radial_profile, sim_r_centers = estimator.radial_profile(sim_intensity)

#     # Display simulated diffraction pattern
#     st.subheader("Simulated Diffraction Pattern")
#     st.image(sim_intensity, caption="Simulated Intensity at Sensor", use_column_width=True, clamp=True)

#     # Plot overlaid radial profiles
#     fig1, ax1 = plt.subplots()
#     ax1.plot(r_centers, exp_radial_profile, label="Experimental")
#     ax1.plot(sim_r_centers, sim_radial_profile, label="Simulated", linestyle="--")
#     ax1.set_xlabel("Radius (µm)")
#     ax1.set_ylabel("Intensity")
#     ax1.set_title("Radial Intensity Profiles")
#     ax1.legend()
#     st.pyplot(fig1)

#     # Plot optimization cost curve
#     fig2, ax2 = plt.subplots()
#     ax2.plot(n_values, costs)
#     ax2.axvline(best_n, color='r', linestyle='--', label=f"Best n = {best_n:.5f}")
#     ax2.set_xlabel("Refractive Index")
#     ax2.set_ylabel("Cost")
#     ax2.set_title("Optimization Cost vs Refractive Index")
#     ax2.legend()
#     st.pyplot(fig2)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom
from laser_holography_simulation import RefractiveIndexEstimator

def extract_radial_profile(image):
    center = tuple(np.array(image.shape) // 2)
    Y, X = np.indices(image.shape)
    R = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    R = R.astype(np.int32)
    
    radial_profile = np.bincount(R.ravel(), image.ravel()) / np.bincount(R.ravel())
    return radial_profile

def main():
    st.title("Laser Holography Simulation")

    # User inputs
    particle_diameter = st.number_input("Particle Diameter (µm)", min_value=1.0, max_value=100.0, value=25.2) * 1e-6
    medium_index = st.number_input("Medium Refractive Index", min_value=1.0, max_value=2.0, value=1.33)
    defocus_distance = st.number_input("Defocus Distance (µm)", min_value=10.0, max_value=1000.0, value=130.0) * 1e-6
    wavelength = st.number_input("Laser Wavelength (nm)", min_value=400.0, max_value=800.0, value=632.8) * 1e-9
    pixel_size = st.number_input("Pixel Size in Object Plane (µm)", min_value=0.01, max_value=10.0, value=0.2579) * 1e-6
    object_na = st.number_input("Object Numerical Aperture", min_value=0.1, max_value=1.5, value=0.75)

    uploaded_file = st.file_uploader("Upload Experimental Image", type=["png", "jpg", "jpeg"])

    if st.button("Run Simulation"):
        # Run simulation
        simulator = RefractiveIndexEstimator(
              sphere_diameter=particle_diameter,
              n_medium=medium_index,
              defocus_distance=defocus_distance,
              wavelength=wavelength,
              object_pixel_size=pixel_size,
              objective_NA=object_na
        )

        # simulator = RefractiveIndexEstimator(
        #     particle_diameter=particle_diameter,
        #     medium_index=medium_index,
        #     defocus_distance=defocus_distance,
        #     wavelength=wavelength,
        #     pixel_size=pixel_size,
        #     object_na=object_na
        #)
        simulated_image = simulator.simulate()

        # Display simulated image
        st.subheader("Simulated Hologram")
        fig1, ax1 = plt.subplots()
        ax1.imshow(simulated_image, cmap='gray')
        ax1.set_title("Simulated Image")
        ax1.axis('off')
        st.pyplot(fig1)

        # Show experimental image if uploaded
        if uploaded_file is not None:
            exp_image = np.array(Image.open(uploaded_file).convert('L'))
            if exp_image.shape != simulated_image.shape:
                exp_image = zoom(exp_image, simulated_image.shape[0]/exp_image.shape[0])

            st.subheader("Experimental Image")
            fig2, ax2 = plt.subplots()
            ax2.imshow(exp_image, cmap='gray')
            ax2.set_title("Experimental Image")
            ax2.axis('off')
            st.pyplot(fig2)

            # Plot radial profiles overlayed
            sim_profile = extract_radial_profile(simulated_image)
            exp_profile = extract_radial_profile(exp_image)

            st.subheader("Radial Intensity Profiles")
            fig3, ax3 = plt.subplots()
            ax3.plot(sim_profile, label="Simulated")
            ax3.plot(exp_profile, label="Experimental")
            ax3.set_xlabel("Radius (pixels)")
            ax3.set_ylabel("Intensity")
            ax3.legend()
            ax3.set_title("Radial Intensity Comparison")
            st.pyplot(fig3)
        else:
            st.info("Please upload an experimental image to compare profiles.")

if __name__ == "__main__":
    main()
