import time
import numpy as np
from multiprocessing import shared_memory
from multiprocessing.connection import Listener

ADDRESS = ("localhost", 9000)
AUTHKEY = b"secret"

def generate_sample():
    """
    Generate a variable-length sequence of numpy arrays.
    """
    num_arrays = np.random.randint(2, 6)
    arrays = []
    for _ in range(num_arrays):
        length = np.random.randint(100, 500)
        arrays.append(np.random.randn(length).astype(np.float32))
    return arrays

def main():
    listener = Listener(ADDRESS, authkey=AUTHKEY)
    print("Process A: waiting for consumer...")
    conn = listener.accept()
    print("Process A: connected")

    while True:

        np_arrays = generate_sample()

        total_bytes = sum(a.nbytes for a in np_arrays)
        shm = shared_memory.SharedMemory(create=True, size=total_bytes)

        metadata = []
        offset = 0

        # Write arrays contiguously
        for arr in np_arrays:
            buf = np.ndarray(arr.shape, dtype=arr.dtype,
                             buffer=shm.buf, offset=offset)
            buf[:] = arr
            metadata.append({
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "offset": offset,
                "nbytes": arr.nbytes,
            })
            offset += arr.nbytes

        # Send control message
        conn.send({
            "shm_name": shm.name,
            "arrays": metadata,
        })

        print(f"Process A: sent sample with {len(metadata)} arrays")

        # IMPORTANT: producer keeps shm alive until consumer is done
        ack = conn.recv()
        if ack == "done":
            shm.close()
            shm.unlink()

        time.sleep(0.1)  # async-ish generation

if __name__ == "__main__":
    main()
