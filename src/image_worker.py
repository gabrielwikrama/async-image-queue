import multiprocessing as mp
import os
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageFilter


def process_single_image(
    input_path: str, output_dir: str, size: Tuple[int, int] = (800, 800)
) -> str:
    """
    Memproses satu gambar secara CPU-bound:
    1. Resize ke 800x800 piksel
    2. Konversi ke Grayscale
    3. Terapkan filter Sharpen
    4. Simpan ke folder output
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = Path(input_path).name
    output_path = os.path.join(output_dir, f"proc_{filename}")

    with Image.open(input_path) as img:
        resized = img.resize(size, Image.Resampling.LANCZOS)
        gray = resized.convert("L")
        sharpened = gray.filter(ImageFilter.SHARPEN)
        sharpened.save(output_path, quality=85)

    return output_path


def worker_process(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    output_dir: str,
    size: Tuple[int, int] = (800, 800),
) -> None:
    """
    Proses worker paralel yang terus mengambil tugas dari multiprocessing.Queue
    hingga menerima sinyal berhenti (None).
    """
    while True:
        image_path = task_queue.get()
        if image_path is None:
            break

        try:
            out_path = process_single_image(image_path, output_dir, size)
            result_queue.put(("OK", out_path))
        except Exception as exc:
            result_queue.put(("ERR", f"{image_path}: {exc}"))


# Uji coba mandiri modul multiprocessing worker
def _test_worker() -> None:
    raw_dir = "dataset/test_raw"
    out_dir = "dataset/test_out"
    os.makedirs(raw_dir, exist_ok=True)

    # Buat 6 gambar dummy untuk pengujian
    sample_paths = []
    for i in range(1, 7):
        p = os.path.join(raw_dir, f"sample_{i:02d}.jpg")
        Image.new("RGB", (1200, 1200), color=(i * 35, 100, 180)).save(p)
        sample_paths.append(p)

    task_q: mp.Queue = mp.Queue()
    result_q: mp.Queue = mp.Queue()
    num_workers = 2

    # Jalankan 2 proses worker paralel
    workers = [
        mp.Process(target=worker_process, args=(task_q, result_q, out_dir))
        for _ in range(num_workers)
    ]
    for w in workers:
        w.start()

    # Masukkan tugas dan sinyal berhenti (None) untuk tiap worker
    for path in sample_paths:
        task_q.put(path)
    for _ in range(num_workers):
        task_q.put(None)

    for _ in range(len(sample_paths)):
        status, info = result_q.get()
        print(f"[{status}] Selesai diproses: {info}")

    for w in workers:
        w.join()

    print("Pengujian Multiprocessing Worker selesai!")


if __name__ == "__main__":
    mp.freeze_support()
    _test_worker()