import asyncio
from typing import Any, Optional


class Node:
    """Representasi satu simpul (node) dalam struktur data Linked Queue."""

    def __init__(self, data: Any) -> None:
        self.data: Any = data
        self.next: Optional["Node"] = None


class AsyncLinkedQueue:
    """
    Implementasi struktur data Queue (FIFO) berbasis Linked List
    dengan operasi enqueue() dan dequeue() asinkron (non-blocking).
    """

    def __init__(self, maxsize: Optional[int] = None) -> None:
        self._head: Optional[Node] = None
        self._tail: Optional[Node] = None
        self._size: int = 0
        self._maxsize: Optional[int] = maxsize
        self._closed: bool = False

        # Sinkronisasi coroutine agar aman dari race condition
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._not_full = asyncio.Condition(self._lock)

    def is_empty(self) -> bool:
        return self._size == 0

    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    async def enqueue(self, item: Any) -> None:
        """Menambahkan item ke ujung belakang (tail) antrian secara asinkron."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("Tidak dapat melakukan enqueue pada antrian yang sudah ditutup.")

            while self._maxsize is not None and self._size >= self._maxsize:
                await self._not_full.wait()

            new_node = Node(item)
            if self._tail is None:
                self._head = self._tail = new_node
            else:
                self._tail.next = new_node
                self._tail = new_node

            self._size += 1
            self._not_empty.notify()

    async def dequeue(self) -> Optional[Any]:
        """
        Mengambil dan menghapus item dari ujung depan (head) antrian secara asinkron.
        Mengembalikan None jika antrian sudah ditutup dan seluruh item habis diproses.
        """
        async with self._lock:
            while self._head is None and not self._closed:
                await self._not_empty.wait()

            if self._head is None and self._closed:
                return None

            removed_node = self._head
            self._head = self._head.next
            if self._head is None:
                self._tail = None

            self._size -= 1
            self._not_full.notify()
            return removed_node.data

    async def close(self) -> None:
        """Menutup antrian setelah seluruh tugas selesai dimasukkan."""
        async with self._lock:
            self._closed = True
            self._not_empty.notify_all()


# Uji coba mandiri untuk memastikan sifat FIFO dan non-blocking berjalan benar
async def _test_queue() -> None:
    queue = AsyncLinkedQueue()

    async def producer() -> None:
        for i in range(1, 6):
            task_name = f"gambar_{i:03d}.jpg"
            await queue.enqueue(task_name)
            print(f"[Enqueue] Masuk: {task_name} | Ukuran antrian: {len(queue)}")
            await asyncio.sleep(0.1)
        await queue.close()

    async def consumer() -> None:
        while True:
            item = await queue.dequeue()
            if item is None:
                break
            print(f"[Dequeue] Diproses: {item} | Sisa antrian: {len(queue)}")
            await asyncio.sleep(0.2)

    await asyncio.gather(producer(), consumer())
    print("Pengujian AsyncLinkedQueue selesai!")


if __name__ == "__main__":
    asyncio.run(_test_queue())