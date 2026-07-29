#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8
r"""
[ADD MODULE DOCUMENTATION HERE]

# ========================================================================== #
#  _  __   _   _                                          __        ___   _  #
# | |/ /  | | | |  Author: Jordan Kuan-Hsien Wu           \ \      / / | | | #
# | ' /   | |_| |  E-mail: jordankhwu@gmail.com            \ \ /\ / /| | | | #
# | . \   |  _  |  Github: https://github.com/JordanWu1997  \ V  V / | |_| | #
# |_|\_\  |_| |_|  Datetime: 2026-07-30 01:27:25             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import datetime
import multiprocessing
import queue
import sys
import time
from multiprocessing.resource_tracker import unregister
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np


def rtsp_producer(rtsp_url, shared_queue, stop_event, live_display=False):
    print("[Producer] Started.")

    # Initialize video capture
    cap = cv2.VideoCapture(rtsp_url)

    # Main loop
    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Retry in 1 secs")
            time.sleep(1)
            continue

        if frame is None:
            time.sleep(0.01)
            continue

        # Timestamp
        timestamp = datetime.datetime.now()

        # Display the video frame
        if live_display:
            cv2.imshow("[Producer] RTSP Stream", frame)
            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # SHM
        shm = None
        try:
            # 1. Dynamically allocate a shared memory block for this frame
            shm = SharedMemory(create=True, size=frame.nbytes)

            # Tell Python's resource tracker to stop watching this block
            # (We use shm._name because the tracker uses the internal POSIX name)
            unregister(shm._name, 'shared_memory')

            # 2. Create a NumPy array that is backed by this shared memory
            shm_array = np.ndarray(frame.shape,
                                   dtype=frame.dtype,
                                   buffer=shm.buf)

            # 3. Copy the OpenCV frame data into the shared memory
            shm_array[:] = frame[:]

            # 4. Put ONLY the metadata in the shared_queue
            metadata = {
                "shm_name": shm.name,
                "shape": frame.shape,
                "dtype": str(frame.dtype),
                "timestamp": timestamp
            }
            shared_queue.put(metadata)
            print(f"[Producer] Sent frame via SHM: {shm.name}")

        except Exception as e:
            if shm is not None:
                shm.unlink()
            print(f'[Producer] Error: {e}')
            break

        finally:
            if shm is not None:
                # Close the producer's access to this block (does NOT delete it)
                shm.close()

        time.sleep(0.01)

    cap.release()
    if live_display:
        cv2.destroyAllWindows()

    print("[Producer] Exited cleanly.")


def inference_consumer(shared_queue, stop_event, live_display=True):
    print("[Consumer] Started.")

    while not stop_event.is_set():
        try:
            # 1. Get the metadata from the shared_queue
            task = shared_queue.get(timeout=0.1)
            shm_name = task["shm_name"]
            shape = task["shape"]
            dtype = task["dtype"]

            shm = None
            try:
                # 2. Connect to the existing shared memory block by name
                shm = SharedMemory(name=shm_name)

                # 3. Reconstruct the NumPy array from the shared memory
                shm_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

                # 4. Process the frame
                print(
                    f"[Consumer] Processed frame {task['shm_name']} from SHM. Shape: {shm_array.shape}"
                )
                canvas = shm_array.copy()
                if live_display:
                    cv2.imshow('[Consumer] Canvas', canvas)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
            finally:
                # 5. CLEANUP: This is critical to prevent RAM leaks
                if shm is not None:
                    shm.close()  # Close access
                    shm.unlink()  # Instruct the OS to DELETE the memory block

        except queue.Empty:
            time.sleep(0.01)
            continue

        except Exception as e:
            print(f"[Consumer] Error: {e}")
            break

    if live_display:
        cv2.destroyAllWindows()

    print("[Consumer] Exited cleanly.")


if __name__ == "__main__":

    # This prevents the executable from infinitely recursively spawning itself
    multiprocessing.freeze_support()

    # Shared Queue
    shared_queue = multiprocessing.Queue()

    # Shared Stop Event
    stop_event = multiprocessing.Event()

    rtsp_url = sys.argv[1] if len(sys.argv) > 1 else 0
    p = multiprocessing.Process(target=rtsp_producer,
                                args=(rtsp_url, shared_queue, stop_event),
                                kwargs={'live_display': False})
    c = multiprocessing.Process(target=inference_consumer,
                                args=(shared_queue, stop_event),
                                kwargs={'live_display': True})

    # Start process
    p.start()
    c.start()

    try:
        # Keep main thread alive while workers do their job
        while p.is_alive() and c.is_alive():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print(
            "\n[Main] Caught KeyboardInterrupt. Initiating graceful shutdown..."
        )
        stop_event.set()  # Signal processes to stop looping

    finally:
        # Wait for processes to exit gracefully
        p.join()
        c.join()

        # DRAIN THE QUEUE
        # Any frames stuck in the queue when the consumer stopped will leak memory.
        # We must pull them out and unlink them manually.
        print("[Main] Draining queue to clean up unread shared memory...")
        cleanup_count = 0
        while not shared_queue.empty():
            try:
                task = shared_queue.get_nowait()
                orphaned_shm = SharedMemory(name=task["shm_name"])
                orphaned_shm.close()
                orphaned_shm.unlink()
                cleanup_count += 1
            except Exception:
                pass

        print(f"[Main] Cleaned up {cleanup_count} orphaned memory blocks.")

        # Explicitly close the processes (Python 3.7+)
        p.close()
        c.close()
        print("[Main] Shutdown complete.")
