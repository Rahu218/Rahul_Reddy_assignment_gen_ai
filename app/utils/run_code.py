import os
import sys
import tempfile
import uuid
import subprocess
from pathlib import Path

import json

def python_safe_path(p):
    # Convert to forward slashes — works perfectly cross-platform in Python
    p = p.replace("\\", "/")
    return p


def run_viz_code_and_save(code_str, output_dir="charts"):
    """
    Runs an LLM-generated matplotlib script safely in a subprocess and moves the produced image
    into `output_dir`. Works cross-platform (Windows, Linux, macOS).

    Assumptions:
      - The LLM-generated code saves the chart to '/tmp/chart.png' (or "/tmp/chart.png").
        This function replaces such references with a real temp path on the current OS.
      - The generated code is simple and doesn't require extra packages beyond pandas/matplotlib.

    Returns:
      - full path to saved image (str) on success
      - None on failure
    """

    # ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # platform temp image path (use a stable filename so generated script can write to it)
    tmpdir = tempfile.gettempdir()
    tmp_image = os.path.join(tmpdir, f"chart_{uuid.uuid4().hex}.png")
    safe_path = python_safe_path(tmp_image)

    # Replace common occurrences of '/tmp/chart.png' in the generated code
    # (handle double and single quotes)
    code_modified = code_str.replace('"/tmp/chart.png"', f'"{safe_path}"')
    code_modified = code_modified.replace("'/tmp/chart.png'", f"'{safe_path}'")
    code_modified = code_modified.replace("/tmp/chart.png", safe_path)

    # Create a secure temporary script file in the OS temp dir
    # Use suffix .py and delete=False so subprocess can read it after we close it
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=tmpdir, encoding="utf-8") as tf:
        script_path = tf.name
        tf.write(code_modified)

    try:
        # Use the same python interpreter that's running this process for better env consistency
        python_exec = sys.executable or "python"
        proc = subprocess.run(
            [python_exec, script_path],
            capture_output=True,
            text=True,
            timeout=30  # adjust if some scripts need more time
        )
    except Exception as exc:
        print("Failed to run script:", exc)
        # cleanup temp script
        try:
            os.remove(script_path)
        except Exception:
            pass
        return None

    # Remove the temporary script (we no longer need it)
    try:
        os.remove(script_path)
    except Exception:
        pass

    if proc.returncode != 0:
        print("Visualization script exited with non-zero status.")
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
        return None

    # Check the temp image exists
    if not os.path.exists(tmp_image):
        print("Script ran but did not produce the image at", tmp_image)
        return None

    # Move to final destination with unique name
    final_name = f"chart_{uuid.uuid4().hex}.png"
    final_path = os.path.join(output_dir, final_name)
    try:
        # On Windows, os.replace will overwrite if exists; use rename for atomic move where possible
        os.replace(tmp_image, final_path)
    except Exception:
        # fallback: copy then delete
        import shutil
        shutil.copy(tmp_image, final_path)
        try:
            os.remove(tmp_image)
        except Exception:
            pass

    return final_path
