from pathlib import Path


def main() -> None:
    benchmarks_dir = Path(__file__).resolve().parent / "benchmarks"
    qasm_count = len(list(benchmarks_dir.rglob("*.qasm"))) if benchmarks_dir.exists() else 0
    print(f"Benchmark placeholder: found {qasm_count} .qasm files in {benchmarks_dir}.")
    print("Next step: integrate Qiskit SABRE transpilation and compare SWAP counts.")


if __name__ == "__main__":
    main()
