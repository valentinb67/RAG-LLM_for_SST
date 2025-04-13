import os
import subprocess
import sys
import time
import argparse

def run_command(command, working_dir=None):
    """Run a shell command and exit on failure."""
    print(f"\nRunning: {command}")
    result = subprocess.run(command, shell=True, cwd=working_dir)
    if result.returncode != 0:
        print(f"Error occurred while running: {command}")
        sys.exit(1)

def wait_for_file(filepath, timeout=30):
    """Wait for a file to be created before proceeding."""
    print(f" Waiting for file: {filepath}")
    start_time = time.time()
    while not os.path.exists(filepath):
        if time.time() - start_time > timeout:
            print(f"Timeout: {filepath} not found after {timeout} seconds.")
            sys.exit(1)
        time.sleep(1)
    print(f"File found: {filepath}")

def extract_data():
    """Step 1: Extract and clean data from PDFs."""
    print("\n[Step 1] Extracting and cleaning PDF data...")
    script_path = os.path.join("src", "data_processing", "hybrid_data_process.py")
    run_command(f"python {script_path}")

def build_vector_index():
    """Step 2: Build FAISS index and save metadata for hybrid retrieval."""
    print("\n[Step 2] Building FAISS hybrid index...")
    script_path = os.path.join("src", "model_management", "hybrid_vector_store.py")
    run_command(f"python {script_path}")

def run_retrieval():
    """Step 3: Test retrieval logic (optional check)."""
    print("\n[Step 3] Testing hybrid retrieval module...")
    script_path = os.path.join("src", "model_management", "hybrid_retrieval.py")
    run_command(f"python {script_path}")

def start_api():
    """Step 4: Start FastAPI backend server with correct PYTHONPATH."""
    print("\n[Step 4] Starting FastAPI server ...")

    # Add src to PYTHONPATH
    src_path = os.path.join(os.path.dirname(__file__), "src")
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{src_path}{os.pathsep}{current_pythonpath}"

    api_command = ["uvicorn", "api.api:app", "--reload"]
    return subprocess.Popen(api_command, cwd=src_path)

def start_gradio_ui():
    """Step 5: Start Gradio-based user interface."""
    print("\n[Step 5] Launching Gradio interface...")
    ui_command = ["python", "ui.py"]
    return subprocess.Popen(ui_command, cwd=os.path.join("src", "ui"))

def main(args):
    if args.extract:
        extract_data()
    if args.index:
        build_vector_index()

    # Ensure FAISS index file is present before continuing
    index_path = os.path.join("models", "index", "index_files", "faiss_index")
    wait_for_file(index_path)

    if args.retrieval:
        run_retrieval()

    api_process = None
    ui_process = None

    if args.api:
        api_process = start_api()
        time.sleep(5)  # Give the API time to boot

    if args.ui:
        ui_process = start_gradio_ui()

    if args.api or args.ui:
        print("\n Project is running. Press Ctrl+C to stop.")
        try:
            if ui_process:
                ui_process.wait()
            if api_process:
                api_process.terminate()
        except KeyboardInterrupt:
            print("\n Terminating processes...")
            if ui_process:
                ui_process.terminate()
            if api_process:
                api_process.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrator for Hybrid RAG-SST Pipeline")
    parser.add_argument("--extract", action="store_true", help="Step 1: Extract and clean data from PDFs")
    parser.add_argument("--index", action="store_true", help="Step 2: Build FAISS hybrid index")
    parser.add_argument("--retrieval", action="store_true", help="Step 3: Run retrieval module (test)")
    parser.add_argument("--api", action="store_true", help="Step 4: Start FastAPI server")
    parser.add_argument("--ui", action="store_true", help="Step 5: Launch Gradio UI")

    # Default behavior: run all
    if len(sys.argv) == 1:
        sys.argv += ["--extract", "--index", "--retrieval", "--api", "--ui"]

    args = parser.parse_args()
    main(args)




