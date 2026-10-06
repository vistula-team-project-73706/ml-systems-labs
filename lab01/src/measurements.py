import statistics
import time
import threading
from pathlib import Path

import joblib
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


RESULTS_DIR = Path("lab01/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROCESS = psutil.Process()


# --------------------------------------------------
# Dataset
# --------------------------------------------------

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)


# --------------------------------------------------
# Model factories
# --------------------------------------------------

def create_logistic():
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
    )


def create_random_forest():
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )


# --------------------------------------------------
# Peak memory monitor
# --------------------------------------------------

def monitor_peak_memory(stop_event, result):
    peak = 0.0

    while not stop_event.is_set():
        memory_mb = PROCESS.memory_info().rss / (1024 * 1024)

        if memory_mb > peak:
            peak = memory_mb

        time.sleep(0.005)

    # Final measurement
    memory_mb = PROCESS.memory_info().rss / (1024 * 1024)

    if memory_mb > peak:
        peak = memory_mb

    result[0] = peak


def measure_peak_memory(function):
    result = [0.0]
    stop_event = threading.Event()

    monitor = threading.Thread(
        target=monitor_peak_memory,
        args=(stop_event, result),
    )

    monitor.start()

    try:
        function()
    finally:
        stop_event.set()
        monitor.join()

    return result[0]


# --------------------------------------------------
# Training time
# --------------------------------------------------

def measure_training_time(model_factory):

    # Warm-up
    model = model_factory()
    model.fit(X_train, y_train)

    times = []

    for _ in range(5):
        model = model_factory()

        start = time.perf_counter()
        model.fit(X_train, y_train)
        end = time.perf_counter()

        times.append(end - start)

    return statistics.median(times)


# --------------------------------------------------
# Inference latency
# --------------------------------------------------

def measure_inference_latency(model):

    sample = X_test[0].reshape(1, -1)

    # Warm-up
    model.predict(sample)

    times = []

    for _ in range(100):
        start = time.perf_counter()
        model.predict(sample)
        end = time.perf_counter()

        times.append((end - start) * 1000)

    return statistics.median(times)


# --------------------------------------------------
# Training peak memory
# --------------------------------------------------

def measure_training_memory(model_factory):

    def train():
        model = model_factory()
        model.fit(X_train, y_train)

    return measure_peak_memory(train)


# --------------------------------------------------
# Inference peak memory
# --------------------------------------------------

def measure_inference_memory(model):

    sample = X_test[0].reshape(1, -1)

    def infer():
        for _ in range(100):
            model.predict(sample)

    return measure_peak_memory(infer)


# --------------------------------------------------
# Model size
# --------------------------------------------------

def save_and_measure_model(model, filename):

    path = RESULTS_DIR / filename

    joblib.dump(model, path)

    size_bytes = path.stat().st_size
    size_kb = size_bytes / 1024

    return size_bytes, size_kb


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    models = [
        (
            "LogisticRegression",
            create_logistic,
            "logistic_regression.joblib",
        ),
        (
            "RandomForest",
            create_random_forest,
            "random_forest.joblib",
        ),
    ]

    results = []

    for name, factory, filename in models:

        print()
        print(f"Measuring {name}...")

        training_time = measure_training_time(factory)

        training_memory = measure_training_memory(factory)

        model = factory()
        model.fit(X_train, y_train)

        inference_latency = measure_inference_latency(model)

        inference_memory = measure_inference_memory(model)

        model_size_bytes, model_size_kb = save_and_measure_model(
            model,
            filename,
        )

        result = {
            "model": name,
            "training_median_seconds": training_time,
            "inference_median_ms": inference_latency,
            "model_size_bytes": model_size_bytes,
            "model_size_kb": model_size_kb,
            "training_peak_memory_mb": training_memory,
            "inference_peak_memory_mb": inference_memory,
        }

        results.append(result)

        print(f"Training median:       {training_time:.6f} seconds")
        print(f"Inference median:      {inference_latency:.6f} ms")
        print(f"Model size:            {model_size_kb:.2f} KB")
        print(f"Training peak memory:  {training_memory:.2f} MB")
        print(f"Inference peak memory: {inference_memory:.2f} MB")

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    output_file = RESULTS_DIR / "measurements.csv"

    headers = [
        "model",
        "training_median_seconds",
        "inference_median_ms",
        "model_size_bytes",
        "model_size_kb",
        "training_peak_memory_mb",
        "inference_peak_memory_mb",
    ]

    with open(output_file, "w") as file:

        file.write(",".join(headers) + "\n")

        for result in results:

            file.write(
                ",".join(
                    str(result[header])
                    for header in headers
                )
                + "\n"
            )

    print()
    print("Measurements saved to:")
    print(output_file)


if __name__ == "__main__":
    main()
