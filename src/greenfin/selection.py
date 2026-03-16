from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from .dataset import GreenFinDataset


def select_farthest_ids(
    ids: List[int],
    kv_dict: Dict[int, torch.Tensor],
    k: int,
    start: str = "centroid",
    seed: int = 42,
) -> List[int]:
    valid_ids, embs = [], []
    for hid in ids:
        e = kv_dict.get(int(hid))
        if e is not None:
            valid_ids.append(int(hid))
            embs.append(e.float().cpu().numpy())
    h = len(valid_ids)
    if h == 0:
        return []
    if h <= k:
        return valid_ids

    e_arr = np.asarray(embs, dtype=np.float32)
    e_arr /= np.linalg.norm(e_arr, axis=1, keepdims=True) + 1e-9

    if start == "centroid":
        c = e_arr.mean(axis=0, keepdims=True)
        c /= np.linalg.norm(c) + 1e-9
        dots = (e_arr @ c.T).squeeze(-1)
        first_idx = int(np.argmax(dots))
    else:
        rng = np.random.RandomState(seed)
        first_idx = int(rng.randint(0, h))

    selected = [first_idx]
    dmin = 1.0 - (e_arr @ e_arr[first_idx : first_idx + 1].T).squeeze(-1)
    dmin[first_idx] = -1.0

    for _ in range(1, k):
        j = int(np.argmax(dmin))
        if dmin[j] < 0:
            break
        selected.append(j)
        new_d = 1.0 - (e_arr @ e_arr[j : j + 1].T).squeeze(-1)
        dmin = np.minimum(dmin, new_d)
        dmin[j] = -1.0

    return [valid_ids[j] for j in selected]


def select_kmeans_ids(
    ids: List[int],
    sents: List[List[float]],
    kv_dict: Dict[int, torch.Tensor],
    k: int,
    batch_size: int = 256,
    max_iter: int = 30,
    random_state: int = 42,
) -> List[int]:
    del sents  # kept for API compatibility with the original code

    valid_ids, embs = [], []
    for hid in ids:
        e = kv_dict.get(int(hid))
        if e is not None:
            valid_ids.append(int(hid))
            embs.append(e.float().cpu().numpy())
    h = len(valid_ids)
    if h <= k:
        return valid_ids

    e_arr = np.asarray(embs, dtype=np.float32)
    e_arr = e_arr / (np.linalg.norm(e_arr, axis=1, keepdims=True) + 1e-9)

    km = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        batch_size=min(batch_size, max(k * 4, 32)),
        max_iter=max_iter,
        random_state=random_state,
        reassignment_ratio=0.02,
        n_init=3,
    )
    labels = km.fit_predict(e_arr)
    centers = km.cluster_centers_

    chosen_ids = []
    for c in range(k):
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            continue
        sub = e_arr[idxs]
        d2 = ((sub - centers[c : c + 1]) ** 2).sum(axis=1)
        j = idxs[np.argmin(d2)]
        chosen_ids.append(valid_ids[int(j)])

    if len(chosen_ids) < k:
        mask = np.ones(h, dtype=bool)
        if chosen_ids:
            sel_idx = [valid_ids.index(hid) for hid in chosen_ids if hid in valid_ids]
            mask[sel_idx] = False
        remaining_idx = np.where(mask)[0]
        if chosen_ids:
            s = e_arr[[valid_ids.index(hid) for hid in chosen_ids]]
            dmin = np.min(1.0 - e_arr[remaining_idx] @ s.T, axis=1)
            order = remaining_idx[np.argsort(-dmin)]
        else:
            order = remaining_idx
        for j in order[: (k - len(chosen_ids))]:
            chosen_ids.append(valid_ids[int(j)])

    return chosen_ids[:k]


def preselect_news(
    ds: GreenFinDataset,
    cap_per_day: Optional[int],
    selection_mode: str = "kmeans",
    seed: int = 42,
    emb_dict: Optional[Dict[int, torch.Tensor]] = None,
) -> None:
    if cap_per_day is None:
        return
    if selection_mode not in ("kmeans", "topconf", "random", "farthest"):
        raise ValueError("selection_mode must be one of: 'kmeans','topconf','random','farthest'")

    rng = np.random.RandomState(seed)

    for s in ds.samples:
        ids_new: List[List[int]] = []
        sents_new: List[List[List[float]]] = []

        for ids_d, sents_d in zip(s.headline_ids, s.sentiments):
            h = len(ids_d)
            if h > cap_per_day:
                if selection_mode == "topconf":
                    p = np.asarray(sents_d, dtype=np.float32)
                    p = np.clip(p, 1e-9, None)
                    p = p / p.sum(axis=1, keepdims=True)
                    ent = -(p * np.log(p)).sum(axis=1)
                    idx = np.argpartition(ent, cap_per_day - 1)[:cap_per_day]
                    idx = np.sort(idx)
                elif selection_mode == "random":
                    idx = np.sort(rng.choice(h, size=cap_per_day, replace=False))
                elif selection_mode == "kmeans":
                    if not emb_dict:
                        idx = np.sort(rng.choice(h, size=cap_per_day, replace=False))
                    else:
                        keep_ids = select_kmeans_ids(
                            ids=ids_d,
                            sents=sents_d,
                            kv_dict=emb_dict,
                            k=cap_per_day,
                            batch_size=256,
                            max_iter=50,
                            random_state=seed,
                        )
                        pos = {int(hid): i for i, hid in enumerate(ids_d)}
                        idx = np.array([pos[int(hid)] for hid in keep_ids if int(hid) in pos], dtype=int)
                        idx.sort()
                else:
                    if not emb_dict:
                        idx = np.sort(rng.choice(h, size=cap_per_day, replace=False))
                    else:
                        keep_ids = select_farthest_ids(
                            ids=ids_d,
                            kv_dict=emb_dict,
                            k=cap_per_day,
                            start="centroid",
                            seed=seed,
                        )
                        pos = {int(hid): i for i, hid in enumerate(ids_d)}
                        idx = np.array([pos[int(hid)] for hid in keep_ids if int(hid) in pos], dtype=int)
                        idx.sort()

                ids_d = [ids_d[i] for i in idx]
                sents_d = [sents_d[i] for i in idx]

            ids_new.append(ids_d)
            sents_new.append(sents_d)

        s.headline_ids = ids_new
        s.sentiments = sents_new
