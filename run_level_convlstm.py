"""NeuroNet prototype + dual-GPU   (v0.3)"""
# ─── Std libs ───────────────────────────────────────────────────────────
import os
import json, threading, asyncio, gc
from pathlib import Path
import shutil
import numpy as np, pandas as pd#, nibabel as nib
#from bids import BIDSLayout       # pip install pybids
#from scipy.ndimage import zoom
import math

# ─── ML libs ────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, mixed_precision, optimizers, losses
from tensorflow.keras.layers import BatchNormalization, ReLU
from monai.transforms import Compose, RandAffine, RandGaussianNoise
from tensorflow_addons.layers import GroupNormalization
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import tensorflow_addons as tfa
        
class LRLogger(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        if batch % 500 == 0:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            print(f"\nstep {batch:>4d}: learning‑rate = {lr:.6f}")

        
# ─── Mixed precision (optional) ─────────────────────────────────────────
mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(False)
 #0. confirm we really are on CPU
print("Visible GPUs =", tf.config.list_physical_devices('GPU'))  #  []

# ─── Constants ──────────────────────────────────────────────────────────
SAVE_DIR   = Path("/home/omkaa/BrainAI")
BATCH_SIZE = 12
VAL_FRAC  = 0.2
RANDOM_SEED = 42
RAW_ROOT  = "/home/omkaa/BrainAI/BIDS_dataset"
DER_ROOT  = "/home/omkaa/BrainAI/BIDS_dataset"
WIN_LEN_S  = 20.0       # window length (e.g. 10 TR = 20 s)
HRF_DELAY  = 6.0        # shift by 6 s (3 TR) to hit the peak
STEP_S     = None       # non-overlapping
META_PATH = Path("/home/omkaa/BrainAI/win_ds/meta.json")
CACHE_DIR  = Path("/home/omkaa/BrainAI/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)   # ← make sure it exists
CACHE_PATH = str(CACHE_DIR / "xrms_v1")        # give TF a **file prefix**

with open(META_PATH) as f:
    meta = json.load(f)

T = meta["T"]                 # 10 or 16
T = int(T)

Z_SCORE_RMS   = True          # per‑volume normalisation
TEMP_LEN      = T             # =16 in your dataset
LR            = 2e-4
WDECAY        = 1e-5          # tiny L2
IMG_SHAPE  = tuple(meta["img_shape"])  # ()
input_shape = (*IMG_SHAPE, 2)
print("using input_shape =", input_shape)
classes    = np.load('/home/omkaa/BrainAI/win_ds/classes.npy', allow_pickle=True)
num_classes = len(classes)

SPLIT_FILE = "/home/omkaa/BrainAI/splits_win_v2.npz"
RAW_SNAPSHOT = "/home/omkaa/BrainAI/win_ds/snapshot"
VAL_FRAC = 0.20
RANDOM_SEED = 42

raw_ds = tf.data.Dataset.load(RAW_SNAPSHOT)

'''rng = np.random.default_rng(RANDOM_SEED)
train_ids, val_ids = [], []
for k in range(num_classes):
    idx = np.where(labels == k)[0]
    rng.shuffle(idx)
    n_val_k = int(round(len(idx) * VAL_FRAC))
    val_ids.append(idx[:n_val_k])
    train_ids.append(idx[n_val_k:])

train_ids = np.concatenate(train_ids).astype(np.int64)
val_ids   = np.concatenate(val_ids).astype(np.int64)

np.savez(SPLIT_FILE,
         seed=RANDOM_SEED,
         val_frac=VAL_FRAC,
         total_windows=N,
         train_ids=train_ids,
         val_ids=val_ids)

print("Saved stratified split:", SPLIT_FILE,
      "| n_train:", train_ids.size, " n_val:", val_ids.size)

print("  | n_train:", len(train_ids), " n_val:", len(val_ids))

# >>> STOP HERE in STEP 1 <<<
import sys
sys.exit(0)'''

'''# Deterministic shuffle once, then split by windows
SHUF_BUFF = min(4096, total_windows)  # or 2048/8192; pick what fits memory
shuffled = raw_ds.shuffle(buffer_size=SHUF_BUFF, seed=RANDOM_SEED,
                          reshuffle_each_iteration=False)
val_raw   = shuffled.take(n_val)
train_raw = shuffled.skip(n_val)'''

npz = np.load(SPLIT_FILE)
assert int(npz["total_windows"]) == int(tf.data.experimental.cardinality(raw_ds).numpy()), \
    "Mismatch: split file was made for a different snapshot/cardinality."

# Ensure no overlap in ids
train_ids_np = npz["train_ids"].astype(np.int64)
val_ids_np   = npz["val_ids"].astype(np.int64)
assert np.intersect1d(train_ids_np, val_ids_np).size == 0, "Train/Val overlap detected."

# fast membership via a lookup table
keys = tf.constant(np.concatenate([train_ids_np, val_ids_np]), tf.int64)
vals = tf.constant(np.concatenate([np.ones_like(train_ids_np), np.zeros_like(val_ids_np)]), tf.int32)
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys, vals),
    default_value=-1
)

base_enum = raw_ds.enumerate()  # (idx, (w,l,s))

def keep_train(i, _): return tf.equal(table.lookup(tf.cast(i, tf.int64)), 1)
def keep_val(i, _):   return tf.equal(table.lookup(tf.cast(i, tf.int64)), 0)

train_raw = base_enum.filter(keep_train).map(lambda i, x: x, num_parallel_calls=tf.data.AUTOTUNE)
val_raw   = base_enum.filter(keep_val).map(lambda i, x: x, num_parallel_calls=tf.data.AUTOTUNE)


def quick_hist(ds, name, limit=20000):
    h = np.zeros(num_classes, dtype=int)
    for i, (_, l, _) in enumerate(ds.as_numpy_iterator()):
        if i >= limit: break
        h[int(l)] += 1
    print(f"[{name}] approx class histogram:", h)

quick_hist(train_raw, "train_raw")
quick_hist(val_raw,   "val_raw")

print("n_train:", len(train_ids_np), " n_val:", len(val_ids_np))

print("train head:", train_ids_np[:10])
print("val head:", val_ids_np[:10])


def one_hot_map(w, lbl, s):
    return w, tf.one_hot(lbl, num_classes), s          # keep subj for filter




def concat_nested(*elems):
    """
    elems : list/tuple of N **identical structures**
            e.g.  [(rmsA, fmA), labelA],
                  [(rmsB, fmB), labelB], ...
    returns : single structure where each corresponding tensor is
              concatenated on axis‑0.

    Effect:
        ((rmsA, fmA), yA)   +
        ((rmsB, fmB), yB)   →   ((rms_batch, fm_batch), y_batch)
    """
    # tf.nest.map_structure walks the tree in parallel
    # (*t) are the N tensors sitting at the same position in each element
    return tf.nest.map_structure(lambda *t: tf.concat(t, axis=0), *elems)

total_windows = int(tf.data.experimental.cardinality(raw_ds).numpy())
print(total_windows)

# ── Helper: build batches with exactly n_per_class windows per class ──

def ds_for_class(xrms_base, k):
    return (xrms_base
            # works whether element is (x,l) or (x,l,s)
            .filter(lambda x, l, *_, kk=tf.constant(k): tf.equal(l, kk))
            # do one-hot here; swallow any extra components
            .map(lambda x, l, *_: (x, tf.one_hot(l, num_classes)),
                 num_parallel_calls=AUTOTUNE)
            .shuffle(256)
            .repeat())

# Compute XRMS AND STD 
def xrms_only(w, lbl, s): #ALSO DOES STD
    # w: (T, D, H, W, 1)
    w = tf.cast(w, tf.float32)

    # Per-voxel aggregates across time (no temporal order)
    mean_t    = tf.reduce_mean(w, axis=0)            # (D,H,W,1)
    mean_sq_t = tf.reduce_mean(tf.square(w), axis=0) # (D,H,W,1)

    rms = tf.sqrt(mean_sq_t)                         # (D,H,W,1)
    var = tf.maximum(mean_sq_t - tf.square(mean_t), 1e-6)
    std = tf.sqrt(var)                               # (D,H,W,1)

    x = tf.concat([rms, std], axis=-1)               # (D,H,W,2)

    # Optional: per-channel z-score across voxels
    if Z_SCORE_RMS:
        mu, v = tf.nn.moments(x, axes=[0,1,2])       # (2,)
        x = (x - mu) / tf.sqrt(v + 1e-6)

    x.set_shape((*IMG_SHAPE, 2))                     # <- static shape
    return x, lbl, s


# If you keep one-hot labels, do it here once (before class filters).
def to_one_hot(x, lbl, s):
    return x, tf.one_hot(lbl, num_classes)

# First: compute XRMS once per window on both splits
AUTOTUNE = tf.data.AUTOTUNE

# --- XRMS with on-disk cache (window-level split) ---
shutil.rmtree(CACHE_DIR, ignore_errors=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---- build split-tagged caches (prevents stale data) ----
import hashlib
def _hash_ids(a: np.ndarray) -> str:
    return hashlib.sha1(a.tobytes()).hexdigest()[:8]

split_tag = f"{_hash_ids(train_ids_np)}_{_hash_ids(val_ids_np)}"
print("[cache] split_tag =", split_tag)

# Optional: wipe *only* old caches for other splits
# for p in CACHE_DIR.glob("xrms_*"):
#     if split_tag not in p.name:
#         p.unlink()

xrms_train = (train_raw
              .map(xrms_only, num_parallel_calls=tf.data.AUTOTUNE)
              .cache(str(CACHE_DIR / f"xrms_train_{split_tag}")))

xrms_val = (val_raw
            .map(xrms_only, num_parallel_calls=tf.data.AUTOTUNE)
            .cache(str(CACHE_DIR / f"xrms_val_{split_tag}")))

print("[cache] Priming XRMS caches (one-time pass)...")
for _ in xrms_train.batch(16).prefetch(tf.data.AUTOTUNE): pass
for _ in xrms_val.batch(16).prefetch(tf.data.AUTOTUNE): pass
print("[cache] Caches are ready.")


# ---- TRAIN branch ---------------------------------------------------
# Per-class streams for balanced batches (same logic as before)

per_cls_train = [ds_for_class(xrms_train, k) for k in range(num_classes)]

def strict_balanced_batches(streams, batch_size, num_classes):
    assert batch_size % num_classes == 0
    n_per_class = batch_size // num_classes
    per_cls_batches = [ds.batch(n_per_class, drop_remainder=True) for ds in streams]
    return tf.data.Dataset.zip(tuple(per_cls_batches)).map(
        lambda *elems: tf.nest.map_structure(lambda *t: tf.concat(t, axis=0), *elems),
        num_parallel_calls=AUTOTUNE
    )

train_ds = (strict_balanced_batches(per_cls_train, BATCH_SIZE, num_classes)
            .shuffle(1024, seed=RANDOM_SEED, reshuffle_each_iteration=True)
            .repeat()                                    # infinite stream
            .prefetch(tf.data.AUTOTUNE))

val_ds = (xrms_val
          .map(lambda x, l, *_: (x, tf.one_hot(l, num_classes)),
               num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))

perm = tf.constant(np.random.permutation(num_classes), dtype=tf.int64)
def map_rand_label(x, l, *_):
    l_shuf = tf.gather(perm, l)
    return x, tf.one_hot(l_shuf, num_classes)

train_ds_rand = (xrms_train
                 .map(lambda x,l,*_: (x,l), num_parallel_calls=AUTOTUNE)
                 .map(map_rand_label, num_parallel_calls=AUTOTUNE)
                 .shuffle(128).repeat()
                 .batch(BATCH_SIZE, drop_remainder=True)
                 .prefetch(AUTOTUNE))
# Compile/fit the same model on train_ds_rand and evaluate on your current val_ds.


# quick sanity check  – one batch should be perfectly balanced

x_chk, y_chk = next(iter(train_ds))
print("batch class histogram ➜", tf.math.bincount(tf.argmax(y_chk, -1),
                                                  minlength=num_classes).numpy())
# should print e.g. [2 2 2 2] for batch_size 8 OR [ 1 1 1 1] FOR BATCH 4


# ─── 2. Build / compile under scope ────────────────────────────────────


def build_rms_cnn_v2(input_shape, n_cls):
    inp = layers.Input(shape=input_shape)        # (D,H,W,2)
    x = inp
    for f, g in [(64, 8), (128, 8), (96, 8)]:
        x = layers.Conv3D(f, 3, strides=2, padding='same', use_bias=False)(x)
        x = GroupNormalization(groups=g, axis=-1)(x)
        x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(n_cls, activation=None, dtype='float32')(x)  # stable head
    return tf.keras.Model(inp, out)




# ─── 3. Dual-GPU strategy ───────────────────────────────────────────────
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1"])
print("[INFO] GPUS in sync:", strategy.num_replicas_in_sync)

alpha = [0.9, .8, 1, 0.8]    # tweak if Cuedts or Stern still lag
loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,           # <- IMPORTANT since the head is linear
    label_smoothing=0.0         # you can try 0.05 later if it helps
)


with strategy.scope():

    model = build_rms_cnn_v2(input_shape, num_classes)
    opt   = tf.keras.optimizers.Adam(learning_rate=LR, weight_decay=WDECAY, clipnorm=1.0)
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy'])  # argmax of logits == argmax of softmax
'''
#     # (A) constant LR 5 e‑4  – easiest to test
    opt = tf.keras.optimizers.Adam(
        learning_rate = 1e-3,          # back to original
        weight_decay  = 1e-5)          # tiny L2

    model = build_rms_cnn((64,64,64,1), num_classes)
    model.add(layers.Dropout(0.3))         # directly after GAP
    model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
#     # (B) or short‑cycle cosine restart each epoch
#     #decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
#     #            initial_learning_rate = 5e-4,
#     #            first_decay_steps     = steps_per_epoch,  # 1‑epoch cycle
#     #            t_mul                 = 1.5,
#     #            alpha                 = 1e-5)
#     #opt = tf.keras.optimizers.Adam(decay, clipnorm=2.0)'''


# ─── 4. Train ───────────────────────────────────────────────────────────


LOG_DIR = "/home/omkaa/BrainAI/tb_logs"
'''vx, vy = next(iter(val_ds.take(1)));  print("val batch:", vx.shape, vy.shape)
tx, ty = next(iter(train_ds.take(1))); print("train batch:", tx.shape, ty.shape)'''

# Rough epoch size (window-level)
n_train = int(train_ids_np.size)
n_val   = int(val_ids_np.size)
steps_per_epoch = max(1, math.ceil(n_train / BATCH_SIZE))

model_tmp = build_rms_cnn_v2(input_shape, num_classes)
model_tmp.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
hist = model_tmp.fit(train_ds_rand, steps_per_epoch=steps_per_epoch, epochs=1,
              validation_data=val_ds,
    verbose=1,                         # force progress bar
    callbacks=[
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda e, logs: print(f"[epoch {e+1}] {logs}")
        ),
    ],
)

print("History keys:", hist.history.keys())
print("val_acc after 1 epoch:", hist.history.get("val_accuracy"))

history = model.fit(
    train_ds,
    epochs=40,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    callbacks=[
        callbacks.ModelCheckpoint("best.keras", monitor="val_accuracy", mode="max", save_best_only=True),
        callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max",
                                    factor=0.5, patience=3, cooldown=1, min_lr=1e-5, verbose=1),
        callbacks.TerminateOnNaN(),
        LRLogger(),
    ],
) 


val_probs = model.predict(val_ds, verbose=0)
val_pred  = val_probs.argmax(1)
print("predicted class histogram on val:", np.bincount(val_pred, minlength=num_classes))
# ─── 5. Save ────────────────────────────────────────────────────────────
SAVE_DIR.mkdir(parents=True, exist_ok=True)
###########################################################################################
# ─── Subject-held-out eval (pure subjects from val_ids) ───────────────────
# ==== STEP 1: CREATE SUBJECT-LEVEL SPLIT (RUN ONCE, THEN COMMENT OUT) ====
# ===== STEP 1 (RUN ONCE): write subject-level split, then exit =====
'''import sys

RAW_SNAPSHOT = "/home/omkaa/BrainAI/win_ds/snapshot"
SPLIT_FILE   = "/home/omkaa/BrainAI/splits_win_v2.npz"  # <- v2 file

raw_ds = tf.data.Dataset.load(RAW_SNAPSHOT)

# subject -> list(window_idx) using snapshot order
subj_to_indices = {}
for idx, (_, _, s) in tf.data.Dataset.enumerate(raw_ds).as_numpy_iterator():
    i = int(idx)
    subj_to_indices.setdefault(s, []).append(i)

subjects = np.array(list(subj_to_indices.keys()), dtype=object)

rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(subjects)

n_val_subj   = max(1, int(round(len(subjects) * VAL_FRAC)))
val_subjects = subjects[:n_val_subj]

val_ids = np.concatenate([np.array(subj_to_indices[s], dtype=np.int64)
                          for s in val_subjects]) if len(val_subjects) else np.array([], np.int64)

total_windows = int(tf.data.experimental.cardinality(raw_ds).numpy())
all_idx   = np.arange(total_windows, dtype=np.int64)
train_ids = np.setdiff1d(all_idx, val_ids, assume_unique=False)

np.savez(
    SPLIT_FILE,
    seed=RANDOM_SEED,
    mode="subject",
    total_windows=total_windows,
    train_ids=train_ids,
    val_ids=val_ids,
    val_subjects=val_subjects,   # bytes
    subjects=subjects            # bytes
)
print("Wrote subject split:", SPLIT_FILE,
      "| #val_subjects:", len(val_subjects),
      "| n_train:", train_ids.size, "| n_val:", val_ids.size)
sys.exit(0)'''


# ─── Subject-held-out eval (using saved subject split) ───────────────────
from pathlib import Path

WEIGHTS = Path(SAVE_DIR) / "best.keras"

if WEIGHTS.exists():
    loaded = tf.keras.models.load_model(
        str(WEIGHTS),
        compile=False,  # <- avoids optimizer loading entirely
        custom_objects={"GroupNormalization": GroupNormalization},
    )
    model.set_weights(loaded.get_weights())
    print(f"[INFO] Loaded FULL model and transferred weights from {WEIGHTS}")
else:
    raise RuntimeError(f"{WEIGHTS} not found. Train first or point to the correct file.")

# Assumes SPLIT_FILE = "/home/omkaa/BrainAI/splits_win_v2.npz" and was created earlier.
SPLIT_FILE   = "/home/omkaa/BrainAI/splits_win_v2.npz"  # <- v2 file
split = np.load(SPLIT_FILE, allow_pickle=True)
print("[split] keys:", split.files)

if "val_subjects" in split.files:
    val_subjects = split["val_subjects"]                     # bytes[]
elif "holdout_subjects" in split.files:
    val_subjects = split["holdout_subjects"]                 # bytes[]
elif "val_ids" in split.files:
    # Derive subjects from window ids
    val_ids = split["val_ids"].astype(np.int64)
    idx_to_subj = {}
    for idx, (_, _, s) in tf.data.Dataset.enumerate(raw_ds).as_numpy_iterator():
        idx_to_subj[int(idx)] = s                            # bytes
    val_subjects = np.unique([idx_to_subj[i] for i in val_ids])
else:
    raise KeyError("SPLIT_FILE lacks val_subjects / holdout_subjects / val_ids")

VAL_SUBJS_T = tf.constant(val_subjects, dtype=tf.string)
print(f"[holdout] #subjects={len(val_subjects)}")

# Build the subject-held-out pipeline: filter → XRMS → one-hot → batch
val_subj_hold = (
    raw_ds
    .filter(lambda w, l, s: tf.reduce_any(tf.equal(s, VAL_SUBJS_T)))
    .map(xrms_only, num_parallel_calls=AUTOTUNE)
    # small eval set → no cache (prevents 'partially cached' warning)
    .map(lambda x, l, *_: (x, tf.one_hot(l, num_classes)), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# Evaluate
val_hold_probs = model.predict(val_subj_hold, verbose=0)
y_hold_true    = np.concatenate([y.numpy() for _, y in val_subj_hold]).argmax(axis=1)
y_hold_pred    = val_hold_probs.argmax(1)

print("predicted class histogram on subject-held-out:",
      np.bincount(y_hold_pred, minlength=num_classes))
print("\nPer-class metrics on subject-held-out:")
print(classification_report(y_hold_true, y_hold_pred, target_names=classes))
print('#################')
# subjects present in train windows
train_subjs = set()
for _, (_, _, s) in tf.data.Dataset.enumerate(train_raw).as_numpy_iterator():
    train_subjs.add(s)
leak = set(val_subjects) & train_subjs
print("subject overlap train∩val:", [s.decode() for s in leak])  # should be []
print('#################')
from sklearn.metrics import accuracy_score, confusion_matrix
print("holdout acc (exact):", accuracy_score(y_hold_true, y_hold_pred))
print("confusion:\n", confusion_matrix(y_hold_true, y_hold_pred))
# majority-class baseline on holdout
print('#################')
maj = np.bincount(y_hold_true).argmax()
print("majority-class acc:", np.mean(y_hold_true == maj))

###########################################################################################
#pd.DataFrame(history.history).to_csv(SAVE_DIR/"convlstm_history.csv",index=False)
print("[INFO] Saved →", SAVE_DIR)

#report clasification metrics 

# 1) Get the ground‑truth labels from the *batched* validation set
y_true = np.concatenate([y.numpy() for _, y in val_ds])
y_true = y_true.argmax(axis=1)          # one‑hot → class index

# 2) Predict probabilities in one shot
y_prob = model.predict(val_ds, verbose=0)
y_pred = y_prob.argmax(axis=1)

# 3) Pretty report
print("\nPer‑class metrics on validation:")
print(classification_report(y_true, y_pred, target_names=classes))

 