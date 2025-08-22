#!/usr/bin/env python
"""
Materialise fMRI windows to a TF-dataset on disk.
Run once, then point the training script at the saved dataset.
"""
# ---------- helper ----------
# ─── Std libs ───────────────────────────────────────────────────────────
import os, json, threading, asyncio, gc, shutil
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from bids import BIDSLayout       # pip install pybids
from scipy.ndimage import zoom

# ─── ML libs ────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, mixed_precision, optimizers,losses
from monai.transforms import Compose, RandAffine, RandGaussianNoise
from tensorflow_addons.layers import GroupNormalization
from sklearn.linear_model import LogisticRegression
import tensorflow_addons as tfa

# ---------- paths & constants ----------
RAW_ROOT   = "/home/omkaa/BrainAI/BIDS_dataset"
DER_ROOT   = RAW_ROOT                      # your derivatives live inside
OUT_DIR    = Path("/home/omkaa/BrainAI/win_ds")     # <— will be created
SNAP = OUT_DIR / "snapshot" 
WIN_LEN_S  = 20.0
HRF_DELAY  = 6.0
STEP_S     = None
IMG_SHAPE  = (64, 64, 64)                  # resample target
TEST_SIZE  = 0.2
RANDOM_SEED = 42
PARENT_TASKS = ["Axcpt", "Cuedts", "Stern", "Stroop"]
label_to_int = {name: idx for idx, name in enumerate(PARENT_TASKS)}
print("label_to_int:", label_to_int)

# ---------------------------------------
       
# ─── Mixed precision (optional) ─────────────────────────────────────────
mixed_precision.set_global_policy("float32")
DER = BIDSLayout(DER_ROOT, validate=False, derivatives=True)
RAW = BIDSLayout(RAW_ROOT, validate=False, derivatives=False)
OUT_DIR.mkdir(exist_ok=True, parents=True)

def count_windows(win_len_s, hrf_delay_s, step_s): #for when window counting is actually needed
    cnt = 0
    for _win, _lbl in gen(win_len_s, hrf_delay_s, step_s):
        cnt += 1
    return cnt

def extract_windows(bold_img, events_df, tr, win_len_s, hrf_delay_s, step_s=None):
    data = bold_img.get_fdata(dtype=np.float32)  # shape (X,Y,Z,T)
    mean_t = data.mean(-1, keepdims=True)
    std_t  = data.std (-1, keepdims=True) + 1e-6
    data   = (data - mean_t) / std_t


    win_len = int(win_len_s / tr)
    hrf_off = int(hrf_delay_s / tr)
    step    = int(step_s / tr) if step_s else win_len

    windows, labels = [], []
    
    for _, ev in events_df.iterrows():
        onset_vol = int(ev.onset / tr) + hrf_off
        end_vol   = onset_vol + win_len
        if end_vol <= data.shape[3]:
            win = data[..., onset_vol:end_vol]             # (X,Y,Z,win_len)
            win = np.expand_dims(win, -1)
            win = zoom(win, (IMG_SHAPE[0]/win.shape[0],
                              IMG_SHAPE[1]/win.shape[1],
                              IMG_SHAPE[2]/win.shape[2],
                              1, 1), order=1)
            win = np.transpose(win, (3, 0, 1, 2, 4))          # (T, 48,48,48,1)
            windows.append(win)
    return np.stack(windows)  # ❹ ***return both***

    
# ---------------------------------------

# ─── 2. Create the tf.data.Dataset and cache ─────────────────────────────
# 1) Before your generator, build the string→int map once:

# ---------- the generator ----------
def gen():
    bold_entries = DER.get(suffix="bold", extension="nii.gz", return_type="object")

    for bfile in bold_entries:
        # 1) Read the per‐run TR from the JSON sidecar
        subj = bfile.entities["subject"] 
        parent =bfile.entities["task"].capitalize()   #  Axcpt / Cuedts / ...

        if parent not in label_to_int:
            continue                                          # ignore other tasks

        # 2) Load the 4D BOLD and its events
        bold_img = nib.load(bfile.path)
        meta = DER.get_metadata(bfile.path)
        tr   = meta["RepetitionTime"]

        # find the matching events in the RAW dataset
        evs = RAW.get(
            subject=bfile.entities['subject'],
            session=bfile.entities['session'],
            task=bfile.entities['task'],
            suffix="events",
            extension="tsv",
            return_type="object"   # or "object"
        )

        if not evs:
            raise FileNotFoundError(f"No events for {bfile.path}")
            
        events_df = pd.read_csv(evs[0].path, sep="\t")
        
        Xw = extract_windows(bold_img,
            events_df,
            tr,                 # per-run TR
            WIN_LEN_S,          # WIN_LEN_S from outer scope
            HRF_DELAY,        # HRF_DELAY from outer scope
            None )             # STEP_S or None
        
        int_lbl = label_to_int[parent]

        for win2 in Xw:
            yield win2, int_lbl, subj.encode()

# -------------------------------------
def main():
    sample_win, _ , subj_str = next(gen())
    T = sample_win.shape[0]            # 10 or 16, whatever the data really is
    IMG_SHAPE = sample_win.shape[1:4]  # already (48,48,48)

#SAVE TIME SO HARD CODE:
    #total_windows = count_windows(WIN_LEN_S, HRF_DELAY, STEP_S)
    #n_val         = int(TEST_SIZE * total_windows)

    total_windows = 13392
    n_val = int(total_windows*TEST_SIZE)
    print(f"Total windows: {total_windows}, val: {n_val}, train: {total_windows-n_val}")

    print("➜  building tf.data.Dataset …")

    # 1) wipe the old snapshot only
    shutil.rmtree(SNAP, ignore_errors=True)
    SNAP.mkdir(parents=True, exist_ok=True)

    ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
            tf.TensorSpec((None, *IMG_SHAPE, 1), tf.float32),  # window
            tf.TensorSpec((), tf.int32),                       # label
            tf.TensorSpec((), tf.string)                       # subject_id
        )
         )

    print("➜  writing dataset to", SNAP)
    SNAP.mkdir(parents=True, exist_ok=True)
    tf.data.Dataset.save(
       ds,
      str(SNAP),               # 3️⃣  one flag = 2×–3× smaller shards
    )

    np.save(OUT_DIR / "classes.npy", np.array(PARENT_TASKS))
    np.save(OUT_DIR / "subjects.npy",
        np.array(list(set(s.decode() for _,_,s in ds.as_numpy_iterator()))))

    with open(OUT_DIR/'meta.json', 'w') as f:
        json.dump(
    {
        "T": T,
        "img_shape": list(IMG_SHAPE),
        "classes": PARENT_TASKS,
        "total_windows": int(tf.data.experimental.cardinality(ds)),
        "dtype": "float32",
        "has_subject_id": True                      # NEW flag
    }, f, indent=2)

    print("✓  done.")

if __name__ == "__main__":        # <-- the guard
    main()