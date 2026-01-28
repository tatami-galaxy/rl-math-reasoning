# process_b.py
import time
import numpy as np
from multiprocessing import shared_memory
from multiprocessing.connection import Client

ADDRESS = ("localhost", 9000)
AUTHKEY = b"secret"

def main():
    conn = Client(ADDRESS, authkey=AUTHKEY)

    while True:
        msg = conn.recv()

        shm = shared_memory.SharedMemory(name=msg["shm_name"])

        arrays = []
        for meta in msg["arrays"]:
            arr = np.ndarray(
                shape=tuple(meta["shape"]),
                dtype=np.dtype(meta["dtype"]),
                buffer=shm.buf,
                offset=meta["offset"],
            )
            arrays.append(np.asarray(arr))  # zero-copy view

        print(f"Process B: received sample with {len(arrays)} arrays")

        # Cleanup
        shm.close()
        conn.send("done")

        # Simulate processing
        print('Processing')
        time.sleep(10)

if __name__ == "__main__":
    main()
