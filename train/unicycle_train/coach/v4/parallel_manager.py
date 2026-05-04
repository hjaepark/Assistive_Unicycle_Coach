# train/unicycle_train/coach/v4/parallel_manager.py

"""
V4 Parallel Manager

Spawns N worker processes, each owning one student.
Provides batched command/result interface to main process.

Uses torch.multiprocessing with "spawn" start method for SAC + PyTorch safety.

Usage (from train.py):
    manager = ParallelManager(
        n_workers=10,
        student_seeds=[1000, 1001, ..., 1009],
        bc_checkpoint=...,
        config=config,
        run_dir=run_dir,
    )
    manager.start()                       # spawn workers, wait for ready

    results = manager.step_all([1, 0, 1, ...])  # send w_t to each, get results
    # results[i] is dict from worker i

    manager.save_all(save_dir)            # save all students
    manager.close()                       # shutdown

Requires caller to ensure torch.multiprocessing.set_start_method("spawn", force=True)
has been called ONCE in main before ParallelManager is instantiated.
"""

import time
from pathlib import Path
from typing import List, Dict, Any

import torch.multiprocessing as mp

from train.unicycle_train.coach.v4.student_worker import worker_main


class ParallelManager:
    def __init__(
        self,
        n_workers: int,
        student_seeds: List[int],
        bc_checkpoint,
        config: dict,
        run_dir: Path,
    ):
        assert len(student_seeds) == n_workers, (
            f"n_workers ({n_workers}) must match len(student_seeds) ({len(student_seeds)})"
        )
        self.n_workers = n_workers
        self.student_seeds = list(student_seeds)
        self.bc_checkpoint = str(bc_checkpoint)
        self.config = config
        self.run_dir = Path(run_dir)

        self.processes: List[mp.Process] = []
        self.pipes = []  # parent ends
        self._started = False

    def start(self, timeout=120.0):
        """Spawn all workers and wait for ready signal from each."""
        ctx = mp.get_context("spawn")

        for i in range(self.n_workers):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            seed = self.student_seeds[i]
            # Each worker logs its env data in a separate subdir
            worker_log_dir = self.run_dir / f"worker_{i:03d}_seed_{seed}"

            p = ctx.Process(
                target=worker_main,
                args=(
                    child_conn,
                    i,                    # worker_id
                    seed,
                    self.bc_checkpoint,
                    self.config,
                    worker_log_dir,
                ),
                daemon=False,
            )
            p.start()
            child_conn.close()  # parent only keeps parent_conn
            self.processes.append(p)
            self.pipes.append(parent_conn)

        # Wait for all workers to signal "ready"
        t0 = time.time()
        ready_count = 0
        for i, pipe in enumerate(self.pipes):
            if pipe.poll(timeout=timeout):
                msg = pipe.recv()
                if msg.get("status") != "ready":
                    raise RuntimeError(
                        f"Worker {i} (seed={self.student_seeds[i]}) failed to init: {msg}"
                    )
                ready_count += 1
            else:
                raise RuntimeError(
                    f"Worker {i} (seed={self.student_seeds[i]}) did not become ready "
                    f"within {timeout}s"
                )

        elapsed = time.time() - t0
        print(f"[ParallelManager] {ready_count}/{self.n_workers} workers ready "
              f"({elapsed:.1f}s)")
        self._started = True

    def step_all(self, w_ts: List[int], timeout=3600.0) -> List[Dict[str, Any]]:
        """
        Send one step command (with w_t) to each worker and collect all results.

        Blocks until all workers return. Returns a list of result dicts in worker order.
        """
        assert self._started, "ParallelManager not started. Call start() first."
        assert len(w_ts) == self.n_workers, (
            f"w_ts length ({len(w_ts)}) must match n_workers ({self.n_workers})"
        )

        # Send step commands
        for i, pipe in enumerate(self.pipes):
            pipe.send(("step", int(w_ts[i])))

        # Collect results (blocking; barrier sync)
        results = [None] * self.n_workers
        for i, pipe in enumerate(self.pipes):
            if not pipe.poll(timeout=timeout):
                raise TimeoutError(
                    f"Worker {i} (seed={self.student_seeds[i]}) did not respond "
                    f"within {timeout}s"
                )
            msg = pipe.recv()
            if msg.get("status") == "error":
                raise RuntimeError(
                    f"Worker {i} (seed={self.student_seeds[i]}) errored:\n"
                    f"{msg.get('traceback', msg.get('error', 'unknown'))}"
                )
            results[i] = msg

        return results

    def save_all(self, save_dir: Path):
        """Ask every worker to save its student under save_dir/student_seed_{seed}.zip"""
        assert self._started, "ParallelManager not started."
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, pipe in enumerate(self.pipes):
            seed = self.student_seeds[i]
            save_path = save_dir / f"student_seed_{seed}"
            pipe.send(("save", str(save_path)))

        for i, pipe in enumerate(self.pipes):
            if pipe.poll(timeout=300.0):
                msg = pipe.recv()
                if msg.get("status") != "saved":
                    print(f"[WARN] Worker {i} save failed: {msg}")
            else:
                print(f"[WARN] Worker {i} save timed out")

    def close(self, save_dir=None, timeout=300.0):
        """Shutdown all workers. Optionally save students before exit."""
        if not self._started:
            return

        save_dir = Path(save_dir) if save_dir is not None else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        # Send close command to all
        for i, pipe in enumerate(self.pipes):
            seed = self.student_seeds[i]
            save_path = None
            if save_dir is not None:
                save_path = str(save_dir / f"student_seed_{seed}_final")
            try:
                pipe.send(("close", save_path))
            except Exception as e:
                print(f"[WARN] Failed to send close to worker {i}: {e}")

        # Wait for acknowledgments
        for i, pipe in enumerate(self.pipes):
            try:
                if pipe.poll(timeout=timeout):
                    pipe.recv()
            except Exception:
                pass

        # Join processes
        for i, p in enumerate(self.processes):
            p.join(timeout=30.0)
            if p.is_alive():
                print(f"[WARN] Worker {i} did not exit cleanly, terminating")
                p.terminate()
                p.join(timeout=5.0)

        # Close pipes
        for pipe in self.pipes:
            try:
                pipe.close()
            except Exception:
                pass

        self._started = False
        print(f"[ParallelManager] {self.n_workers} workers closed")