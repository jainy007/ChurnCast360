import subprocess
import time


def run_step(description, command):
    print(f"\nStarting: {description}")
    start = time.time()
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"Completed: {description} in {time.time() - start:.2f} seconds")
    except subprocess.CalledProcessError as e:
        print(f"Failed: {description}")
        print(e)
        exit(1)


def main():
    print("\nRestore Pipeline: Start\n")
    total_start = time.time()

    steps = [
        ("Data Ingestion", "python src/ingest/load_datasets.py"),
        ("Exploratory Data Analysis", "python src/eda/run_eda.py"),
        (
            "Feature Engineering (with parallelization)",
            "python src/eda/feature_engineering.py",
        ),
        ("Load to SQLite", "python src/sql/load_to_sqlite.py"),
        ("Create Unified View", "python src/sql/create_unified_view.py"),
        ("Train Models", "python src/model/train_tuned_models.py --device auto"),
    ]

    for description, command in steps:
        run_step(description, command)

    print(
        f"\nRestore pipeline completed successfully in {time.time() - total_start:.2f} seconds\n"
    )


if __name__ == "__main__":
    main()
