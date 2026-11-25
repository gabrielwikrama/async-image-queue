import asyncio
import csv
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from image_worker import process_single_image
from orchestrator import AsyncImageBatchOrchestrator


def generate_dummy_dataset(dataset_dir: str, count: int) -> List[str]:
    """Membuat gambar dummy resolusi 1200x1200 untuk kebutuhan pengujian skala besar."""
    os.makedirs(dataset_dir, exist_ok=True)
    paths: List[str] = []

    for i in range(1, count + 1):
        file_path = os.path.join(dataset_dir, f"img_{i:04d}.jpg")
        if not os.path.exists(file_path):
            color = ((i * 37) % 255, (i * 73) % 255, (i * 19) % 255)
            Image.new("RGB", (1200, 1200), color=color).save(file_path, quality=90)
        paths.append(file_path)

    return paths


def run_sequential_baseline(image_paths: List[str], output_dir: str) -> float:
    """Metode pembanding: pemrosesan gambar sekuensial (1 core, sinkron)."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.perf_counter()
    for path in image_paths:
        process_single_image(path, output_dir)
    return time.perf_counter() - start_time


async def run_async_mp_architecture(
    image_paths: List[str], output_dir: str, num_workers: int
) -> float:
    """Arsitektur usulan: AsyncLinkedQueue + Multiprocessing Queue."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    orchestrator = AsyncImageBatchOrchestrator(
        num_workers=num_workers, queue_maxsize=200
    )
    start_time = time.perf_counter()
    await orchestrator.run_batch(image_paths, output_dir)
    return time.perf_counter() - start_time


def save_results_csv(records: List[Dict], csv_path: str) -> None:
    """Menyimpan rekap metrik pengujian ke file CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "jumlah_gambar",
        "metode",
        "waktu_detik",
        "throughput_img_per_detik",
        "speedup",
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def generate_charts(records: List[Dict], results_dir: str) -> None:
    """Membuat grafik perbandingan waktu eksekusi dan throughput untuk makalah IEEE."""
    batch_sizes = sorted(list({r["jumlah_gambar"] for r in records}))
    methods = [
        "Sequential (1 Core)",
        "AsyncQueue + MP (2 Workers)",
        "AsyncQueue + MP (4 Workers)",
        "AsyncQueue + MP (8 Workers)",
    ]

    # 1. Grafik Waktu Eksekusi
    plt.figure(figsize=(8, 5))
    for method in methods:
        times = [
            next(
                r["waktu_detik"]
                for r in records
                if r["jumlah_gambar"] == b and r["metode"] == method
            )
            for b in batch_sizes
        ]
        plt.plot(batch_sizes, times, marker="o", linewidth=2, label=method)

    plt.title("Perbandingan Waktu Eksekusi Pemrosesan Gambar Batch")
    plt.xlabel("Jumlah Gambar (Batch Size)")
    plt.ylabel("Waktu Eksekusi (Detik)")
    plt.xticks(batch_sizes)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "grafik_waktu_eksekusi.png"), dpi=300)
    plt.close()

    # 2. Grafik Throughput
    plt.figure(figsize=(8, 5))
    for method in methods:
        throughputs = [
            next(
                r["throughput_img_per_detik"]
                for r in records
                if r["jumlah_gambar"] == b and r["metode"] == method
            )
            for b in batch_sizes
        ]
        plt.plot(batch_sizes, throughputs, marker="s", linewidth=2, label=method)

    plt.title("Perbandingan Throughput (Gambar per Detik)")
    plt.xlabel("Jumlah Gambar (Batch Size)")
    plt.ylabel("Throughput (Gambar / Detik)")
    plt.xticks(batch_sizes)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "grafik_throughput.png"), dpi=300)
    plt.close()


async def main() -> None:
    raw_dir = "dataset/benchmark_raw"
    out_dir = "dataset/benchmark_out"
    results_dir = "results"

    batch_sizes = [100, 500, 1000]
    worker_configs = [2, 4, 8]
    records: List[Dict] = []

    print("Menyiapkan dataset maksimum 1.000 gambar...")
    all_images = generate_dummy_dataset(raw_dir, max(batch_sizes))

    for count in batch_sizes:
        subset = all_images[:count]
        print(f"\n=== Pengujian Skala {count} Gambar ===")

        # 1. Uji Sequential Baseline
        seq_time = run_sequential_baseline(subset, out_dir)
        seq_tp = count / seq_time
        print(f"Sequential (1 Core)         : {seq_time:.2f} dtk | {seq_tp:.2f} img/dtk")
        records.append(
            {
                "jumlah_gambar": count,
                "metode": "Sequential (1 Core)",
                "waktu_detik": round(seq_time, 4),
                "throughput_img_per_detik": round(seq_tp, 2),
                "speedup": 1.0,
            }
        )

        # 2. Uji AsyncLinkedQueue + Multiprocessing dengan variasi worker
        for w in worker_configs:
            label = f"AsyncQueue + MP ({w} Workers)"
            mp_time = await run_async_mp_architecture(subset, out_dir, num_workers=w)
            mp_tp = count / mp_time
            speedup = seq_time / mp_time
            print(
                f"{label:<28}: {mp_time:.2f} dtk | {mp_tp:.2f} img/dtk | Speedup: {speedup:.2f}x"
            )
            records.append(
                {
                    "jumlah_gambar": count,
                    "metode": label,
                    "waktu_detik": round(mp_time, 4),
                    "throughput_img_per_detik": round(mp_tp, 2),
                    "speedup": round(speedup, 2),
                }
            )

    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    save_results_csv(records, csv_path)
    generate_charts(records, results_dir)
    print(f"\nSelesai! Hasil CSV dan grafik disimpan di folder '{results_dir}/'.")


if __name__ == "__main__":
    mp.freeze_support()
    asyncio.run(main())