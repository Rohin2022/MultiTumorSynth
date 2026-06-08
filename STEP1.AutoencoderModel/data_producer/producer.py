import os

# Must be before ALL other imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

import time
import random
import uuid
import torch
import multiprocessing as mp
import hydra
import json
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import DictConfig

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SpatialPadd,
)
from monai.transforms.transform import MapTransform
from monai.transforms.io.array import LoadImage
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
from copy import deepcopy
import numpy as np
from typing import Optional, Union
from monai.config import DtypeLike, KeysCollection

DEFAULT_POST_FIX = PostFix.meta()


class LoadImaged_BodyMap(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def register(self, reader: ImageReader):
        self._loader.register(reader)

    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            try:
                data = self._loader(d[key], reader)
            except Exception as e:
                print("=" * 80)
                print("FAILED CASE:", d.get("name"))
                print("IMAGE PATH:", d.get(key))
                print("ERROR:", repr(e))
                print("=" * 80)
                raise

            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]

        organ_lbl, meta_information = self.label_transfer(d['segmentations'], d['image'].shape)

        if hasattr(d['image'], "meta"):
            from monai.data.meta_tensor import MetaTensor
            chosen_meta = deepcopy(meta_information) if meta_information is not None else deepcopy(d['image'].meta)
            d['label'] = MetaTensor(organ_lbl, meta=chosen_meta)
        else:
            d['label'] = organ_lbl

        d['label_meta_dict'] = meta_information
        return d

    def label_transfer(self, segmentations_list, shape):
        organ_lbl = np.zeros(shape, dtype=np.uint8)

        organ_mapping = {
            'spleen': 1, 'bladder': 2, 'gall_bladder': 3, 'esophagus': 4,
            'stomach': 5, 'duodenum': 6, 'colon': 7, 'prostate': 8, 'uterus': 9,
            'spleen_lesion': 10, 'bladder_lesion': 11, 'gallbladder_lesion': 12,
            'esophagus_lesion': 13, 'stomach_lesion': 14, 'duodenum_lesion': 15,
            'colon_lesion': 16, 'prostate_lesion': 17, 'uterus_lesion': 18
        }

        def fetch_mask(seg_info):
            file_path = seg_info["file_path"]
            organ_name = seg_info["organ_name"]
            label_idx = organ_mapping[organ_name]
            try:
                array, meta = self._loader(file_path)
                if hasattr(array, "as_tensor"):
                    array_np = array.as_tensor().numpy()
                elif hasattr(array, "numpy"):
                    array_np = array.numpy()
                else:
                    array_np = array
                return label_idx, array_np, meta
            except Exception as e:
                print(f"Failed organ file: {file_path}")
                raise e

        meta_information = None

        with ThreadPoolExecutor(max_workers=18) as executor:
            futures = [executor.submit(fetch_mask, seg) for seg in segmentations_list]
            for future in as_completed(futures):
                label_idx, array_np, meta = future.result()
                if meta_information is None and meta is not None:
                    meta_information = meta
                organ_lbl[array_np > 0] = label_idx

        return organ_lbl, meta_information


def init_worker(cfg: DictConfig):
    global transform_pipeline

    transform_pipeline = Compose([
        LoadImaged_BodyMap(keys=["image"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(cfg.dataset.space_x, cfg.dataset.space_y, cfg.dataset.space_z), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=cfg.dataset.a_min, a_max=cfg.dataset.a_max, b_min=cfg.dataset.b_min, b_max=cfg.dataset.b_max, clip=True),
        SpatialPadd(keys=["image", "label"], spatial_size=(cfg.dataset.roi_x, cfg.dataset.roi_y, cfg.dataset.roi_z), mode=["minimum", "constant"]),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(cfg.dataset.roi_x, cfg.dataset.roi_y, cfg.dataset.roi_z), pos=10, neg=1, num_samples=cfg.producer.num_samples, image_key="image", image_threshold=-1),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        ToTensord(keys=["image", "label"]),
    ])


def disk_writer_thread(write_queue, worker_id, batches_saved, last_save_time, chunk_size):
    """Background thread that continuously pulls batches and writes to disk."""
    while True:
        payload = write_queue.get()
        if payload is None:  # Poison pill
            write_queue.task_done()
            break
        try:
            batch_dict, temp_path, final_path = payload
            torch.save(batch_dict, temp_path)
            os.rename(temp_path, final_path)

            now = time.time()
            with last_save_time.get_lock():
                delta = now - last_save_time.value
                last_save_time.value = now

            samples_per_sec = chunk_size / delta if delta > 0 else 0
            print(
                f"[Worker {worker_id}] Saved {os.path.basename(final_path)} "
                f"| total={batches_saved.value} "
                f"| Δt={delta:.1f}s "
                f"| {samples_per_sec:.0f} samples/s",
                flush=True
            )
        except Exception as e:
            print(f"[Writer {worker_id}] FAILED to save {final_path}: {e}", flush=True)
        finally:
            write_queue.task_done()


def continuous_worker_loop(worker_id, all_data_dicts, cfg, batches_saved, last_save_time):
    # Reinforce thread limits in worker process — ITK re-reads on first use
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

    random.seed(os.getpid() + int(time.time()))
    print(f"[Worker {worker_id}] Started (PID {os.getpid()})", flush=True)

    init_worker(cfg)
    local_batch_buffer = []

    scratch_dir = cfg.producer.scratch_dir
    max_batches = cfg.producer.max_batches
    chunk_size = cfg.producer.chunk_size

    consecutive_failures = 0

    # Non-daemon so it won't be killed before flushing
    write_queue = queue.Queue(maxsize=2)
    writer = threading.Thread(
        target=disk_writer_thread,
        args=(write_queue, worker_id, batches_saved, last_save_time, chunk_size),
        daemon=False
    )
    writer.start()

    def shutdown_writer():
        write_queue.put(None)
        write_queue.join()
        writer.join()

    while True:
        random.shuffle(all_data_dicts)

        for data_dict in all_data_dicts:

            # 1. Termination check
            if random.random() < 0.1:
                if batches_saved.value >= max_batches:
                    print(f"[Worker {worker_id}] Cache target reached. Exiting cleanly.", flush=True)
                    shutdown_writer()
                    return

            # 2. Process data
            try:
                outputs = transform_pipeline(data_dict)
                if not isinstance(outputs, list):
                    outputs = [outputs]

                for crop in outputs:
                    pure_image = crop["image"].as_tensor()
                    pure_label = crop["label"].as_tensor()

                    flattened_label = pure_label.flatten().long()
                    counts = torch.bincount(flattened_label, minlength=19)
                    class_presence = (counts[:19] > 0).to(torch.uint8)

                    local_batch_buffer.append({
                        "image": pure_image.to(torch.float16, copy=True),
                        "label": pure_label.to(torch.int8, copy=True),
                        "contents": class_presence
                    })

                    if worker_id % 24 == 0 and len(local_batch_buffer) % 16 == 0:
                        print(f"[Worker {worker_id}] Buffer filling: {len(local_batch_buffer)} / {chunk_size}", flush=True)

                consecutive_failures = 0

            except Exception as e:
                print("============= FAILED TO LOAD/TRANSFORM =============", flush=True)
                print(f"[Worker {worker_id}] Failed on {data_dict.get('name')}: {e}", flush=True)
                print("====================================================", flush=True)

                consecutive_failures += 1
                if consecutive_failures >= 10:
                    shutdown_writer()
                    raise RuntimeError(f"[Worker {worker_id}] Failed 10 consecutive times. Weka storage might be offline.")
                continue

            # 3. Save when buffer is full
            if len(local_batch_buffer) >= chunk_size:

                with batches_saved.get_lock():
                    if batches_saved.value >= max_batches:
                        print(f"[Worker {worker_id}] Cache met right before write. Discarding buffer and exiting.", flush=True)
                        shutdown_writer()
                        return
                    batches_saved.value += 1
                    current_count = batches_saved.value

                print(f"[Worker {worker_id}] Queuing batch {current_count}/{max_batches} for write...", flush=True)

                process_buffer = local_batch_buffer[:chunk_size]
                local_batch_buffer = local_batch_buffer[chunk_size:]

                batch_dict = {
                    "image": torch.stack([d["image"] for d in process_buffer]),
                    "label": torch.stack([d["label"] for d in process_buffer]),
                    "contents": torch.stack([d["contents"] for d in process_buffer])
                }

                file_id = f"batch_w{worker_id}_{uuid.uuid4().hex[:8]}.pt"
                temp_path = os.path.join(scratch_dir, f"_{file_id}.tmp")
                final_path = os.path.join(scratch_dir, file_id)

                # Blocks if writer is behind — natural backpressure
                write_queue.put((batch_dict, temp_path, final_path))


def start_producer(cfg: DictConfig):
    # Must be called before any mp.Value — fork inherits parent memory,
    # no shared memory gymnastics needed, and works correctly with Hydra
    mp.set_start_method('fork', force=True)

    last_save_time = mp.Value('d', time.time())

    scratch_dir = cfg.producer.scratch_dir
    os.makedirs(scratch_dir, exist_ok=True)

    existing_files = len([f for f in os.listdir(scratch_dir) if f.endswith(".pt")])
    print(f"Found {existing_files} existing .pt files in cache.")

    # mp.Value created AFTER set_start_method
    batches_saved = mp.Value('i', existing_files)

    manifest_path = 'dataset_manifest.json'
    print(f"Reading manifest from {manifest_path}...")
    with open(manifest_path, 'r') as f:
        all_data_dicts = json.load(f)  # loaded once in parent, inherited by fork
    print(f"Loaded {len(all_data_dicts)} data dicts from manifest.")

    num_workers = cfg.producer.num_workers
    print(f"Starting {num_workers} workers writing chunks of {cfg.producer.chunk_size} to {scratch_dir}...")

    workers = []
    for i in range(num_workers):
        p = mp.Process(
            target=continuous_worker_loop,
            args=(i, all_data_dicts, cfg, batches_saved, last_save_time)
        )
        p.start()
        print(f"Spawned worker {i} with PID {p.pid}")
        workers.append(p)

    for p in workers:
        p.join()

    print(f"All workers finished. Total batches saved: {batches_saved.value}")


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig, args=None):
    start_producer(cfg)


if __name__ == "__main__":
    run()