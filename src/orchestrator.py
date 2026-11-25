import asyncio
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Memastikan modul di folder src/ dapat diimpor baik dijalankan langsung maupun sebagai package
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from linked_queue import AsyncLinkedQueue
from image_worker import worker_process


class AsyncImageBatchOrchestrator:
    """
    Orkestrator yang menghubungkan AsyncLinkedQueue (non-blocking ingestion)
    dengan Multiprocessing Queue (CPU-bound parallel image processing).
    """

    def __init__(
        self,
        num_workers: int = 4,
        queue_maxsize: int = 100,
        image_size: Tuple[int, int] = (800, 800),
    ) -> None:
        self.num_workers = num_workers
        self.queue_maxsize = queue_maxsize
        self.image_size = image_size

    async def _producer(
        self, async_queue: AsyncLinkedQueue, image_paths: List[str]
    ) -> None:
        """Memasukkan seluruh path gambar ke AsyncLinkedQueue secara non-blocking."""
        for path in image_paths:
            await async_queue.enqueue(path)
        await async_queue.close()

    async def _dispatcher(
        self, async_queue: AsyncLinkedQueue, mp_task_queue: mp.Queue
    ) -> None:
        """
        Mengambil tugas dari AsyncLinkedQueue (.dequeue) dan meneruskannya
        ke multiprocessing.Queue agar dikerjakan worker paralel.
        """
        loop = asyncio.get_running_loop()
        while True:
            item = await async_queue.dequeue()
            if item is None:
                break
            # Menjalankan mp_task_queue.put di executor agar event loop tidak terblokir
            await loop.run_in_executor(None, mp_task_queue.put, item)

        # Kirim sinyal berhenti (None) ke setiap proses worker
        for _ in range(self.num_workers):
            await loop.run_in_executor(None, mp_task_queue.put, None)

    async def _collector(
        self, mp_result_queue: mp.Queue, total_tasks: int
    ) -> Tuple[int, int]:
        """Mengumpulkan hasil pemrosesan dari seluruh worker tanpa memblokir event loop."""
        loop = asyncio.get_running_loop()
        success_count = 0
        error_count = 0

        for _ in range(total_tasks):
            status, _ = await loop.run_in_executor(None, mp_result_queue.get)
            if status == "OK":
                success_count += 1
            else:
                error_count += 1

        return success_count, error_count

    async def run_batch(
        self, image_paths: List[str], output_dir: str
    ) -> Tuple[int, int]:
        """Menjalankan seluruh pipeline pemrosesan gambar paralel secara end-to-end."""
        os.makedirs(output_dir, exist_ok=True)
        total_tasks = len(image_paths)
        if total_tasks == 0:
            return 0, 0

        async_queue = AsyncLinkedQueue(maxsize=self.queue_maxsize)
        mp_task_queue: mp.Queue = mp.Queue(maxsize=self.num_workers * 4)
        mp_result_queue: mp.Queue = mp.Queue()

        workers = [
            mp.Process(
                target=worker_process,
                args=(mp_task_queue, mp_result_queue, output_dir, self.image_size),
            )
            for _ in range(self.num_workers)
        ]

        for w in workers:
            w.start()

        try:
            _, _, (success_count, error_count) = await asyncio.gather(
                self._producer(async_queue, image_paths),
                self._dispatcher(async_queue, mp_task_queue),
                self._collector(mp_result_queue, total_tasks),
            )
        finally:
            for w in workers:
                w.join()

        return success_count, error_count


# Uji coba integrasi end-to-end
async def _test_orchestrator() -> None:
    raw_dir = "dataset/test_raw"
    out_dir = "dataset/test_orchestrator_out"

    sample_paths = [
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    orchestrator = AsyncImageBatchOrchestrator(num_workers=2, queue_maxsize=10)
    ok, err = await orchestrator.run_batch(sample_paths, out_dir)
    print(f"Integrasi Orchestrator selesai | Berhasil: {ok}, Gagal: {err}")


if __name__ == "__main__":
    mp.freeze_support()
    asyncio.run(_test_orchestrator())