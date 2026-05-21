import os
import subprocess
import tempfile
import numpy as np
import traceback
from typing import Optional

class SAM3DEngine:
    """
    Wrapper for the external SAM3D pipeline.
    Calls a standalone script in a dedicated python environment via subprocess.
    """
    def __init__(self, sam3d_dir: str = "/home/arthur.chiron/sam-3d-objects"):
        self.sam3d_dir = sam3d_dir
        self.venv_name = "sam3d-engine"
        self.script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_inference.py")

    def generate_voxels(self, crop_2d: np.ndarray) -> np.ndarray:
        """
        Runs the external SAM3D pipeline and returns a (64, 64, 64) numpy array.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "input.npy")
            out_path = os.path.join(tmpdir, "output.npy")
            
            # Save the 2D input for the external script
            np.save(in_path, crop_2d)
            
            # Path to the specific python venv containing PyTorch3D and SAM3D deps
            python_path = f"/home/arthur.chiron/.conda/envs/{self.venv_name}/bin/python"
            
            # Construct environment for isolation
            env = os.environ.copy()
            if "PYTHONPATH" in env:
                del env["PYTHONPATH"]
            env["PATH"] = f"/home/arthur.chiron/.conda/envs/{self.venv_name}/bin:{env.get('PATH', '')}"

            cmd = [
                python_path, self.script_path,
                "--input", in_path,
                "--output", out_path
            ]
            
            print(f"[VoCell] Running SAM3D externally via conda env...")
            try:
                # Subprocess call to the venv
                subprocess.run(cmd, check=True, cwd=self.sam3d_dir, env=env)
                
                # Retrieve the generated 3D occupancy grid
                if os.path.exists(out_path):
                    occupancy = np.load(out_path)
                    return occupancy
                else:
                    raise FileNotFoundError(f"SAM3D output {out_path} not found.")
                    
            except subprocess.CalledProcessError as e:
                print(f"[VoCell] Error executing SAM3D script.")
                traceback.print_exc()
                raise e

# --- Singleton Engine Access ---
_GLOBAL_ENGINE: Optional[SAM3DEngine] = None

def get_sam3d_engine() -> SAM3DEngine:
    """Returns a singleton instance of the SAM3D engine."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = SAM3DEngine()
    return _GLOBAL_ENGINE
