import subprocess
import sys
from datetime import datetime

def run_model(script_name):
    """Run a single model script and return its exit code"""
    print(f"Starting {script_name}...")
    try:
        result = subprocess.run([sys.executable, f"scripts/models/{script_name}"], 
                              capture_output=True, 
                              text=True)
        if result.returncode != 0:
            print(f"Error running {script_name}:")
            print(result.stderr)
        else:
            print(f"Successfully completed {script_name}")
        return result.returncode
    except Exception as e:
        print(f"Exception running {script_name}: {str(e)}")
        return 1

def main():
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting sequential model training run at {timestamp}")

    # List of model scripts to run
    model_scripts = [
        "0_knn_classifier.py",
        "1_random_forest.py",
        "2_mlp_nn.py",
        "3_tabnet.py",
        "4_ensemble.py"
    ]

    # Run each model sequentially
    for script in model_scripts:
        result = run_model(script)
        if result != 0:
            print(f"Model {script} failed to complete successfully.")
            sys.exit(1)

    # Run ensemble model
    print("\nAll individual models completed. Running ensemble model...")
    ensemble_result = run_model("4_ensemble.py")
    
    if ensemble_result != 0:
        print("Ensemble model failed to complete successfully.")
        sys.exit(1)
    
    print("\nAll models completed successfully!")

if __name__ == "__main__":
    main() 