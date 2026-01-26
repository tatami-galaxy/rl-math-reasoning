# process_b.py
import time
import numpy as np
import jax.numpy as jnp
from multiprocessing import shared_memory
from multiprocessing.connection import Client

ADDRESS = ("localhost", 9000)
AUTHKEY = b"secret"

def main():
    conn = Client(ADDRESS, authkey=AUTHKEY)
    print("Process B: connected to producer")

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
            arrays.append(jnp.asarray(arr))  # zero-copy view

        print(f"Process B: received sample with {len(arrays)} arrays")

        # Simulate processing
        time.sleep(2)

        # Cleanup
        shm.close()
        conn.send("done")

if __name__ == "__main__":
    main()
