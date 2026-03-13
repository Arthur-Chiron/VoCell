# VoCell : 3D Augmentations for Nuclei

The main purpose of this repository is to generate 3D volumetric augmentations starting from 2D images of cell nuclei. By taking a single 2D slice and reconstructing a pseudo-3D volume (e.g., using Gaussian depth profiles), this tool allows for arbitrary 3D spatial augmentations, complex arbitrary slicing, and cross-sectional sampling to enrich biological datasets.

## 🔬 Nuclei 3D Voxel Explorer

This project includes an interactive 3D visualization dashboard built with Streamlit. It allows you to explore 2D cell nucleus images projected into a 3D voxel space (using a Gaussian depth profile), complete with arbitrary 3D plane slicing, internal masking, and "Minecraft"-style solid cubic rendering.

### Prerequisites

Ensure you have your 2D nuclei crop data correctly placed in the following directory:
`data/CODEX/crops.npy`

### How to Run the Visualization Space

1. **Activate the virtual environment** (if you have one set up):
   ```bash
   source .venv/bin/activate
   ```

2. **Install the required dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app**:
   Run the following command from the root of the project:
   ```bash
   streamlit run src/app.py
   ```

4. **Open the Dashboard**:
   Once the server starts, it will provide a local URL in your terminal (usually `http://localhost:8501`). Open this address in your web browser to interact with the 3D Voxel Explorer!
