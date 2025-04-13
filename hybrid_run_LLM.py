import os
import subprocess
import time

def start_fastapi():
    print("\n [Step 1] Lancement du serveur FastAPI (http://127.0.0.1:8000)...")

    # Path to src/
    src_path = os.path.join(os.path.dirname(__file__), "src")

    # Add src to PYTHONPATH
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{src_path}{os.pathsep}{current_pythonpath}"

    # Uvicorn order
    return subprocess.Popen(["uvicorn", "api.api:app", "--reload"], cwd=src_path)

def start_gradio():
    print("\n Step 2: Launching the Gradio interface ...")

    ui_path = os.path.join("src", "ui")
    return subprocess.Popen(["python", "ui.py"], cwd=ui_path)

def main():
    fastapi_proc = start_fastapi()
    time.sleep(5)  

    gradio_proc = start_gradio()

    print("\n Application ready")
    print(" Ctrl+C to stop")

    try:
        gradio_proc.wait()
    except KeyboardInterrupt:
        print("\n Interruption")
        gradio_proc.terminate()
        fastapi_proc.terminate()

if __name__ == "__main__":
    main()







