# train_jets_cugar_vit_v4.py
# CUGAR ViT-B/16 — v4, built from your v3 run's results: 88.08% single /
# 88.70% TTA (+0.87 pts over v2), Top-5 93.72% (down slightly from 94.50% —
# see note below), macro F1 0.897, hard-class accuracy 87.64%.

import os, re, csv, pickle, hashlib
from collections import Counter, defaultdict
from pathlib import Path

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
import pandas as pd
import numpy as np
import cv2
from PIL import Image

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ============================================================================
# CONFUSION_CLUSTERS — derived from your actual confusion_matrix.csv via
# derive_clusters_from_confusion_csv() (greedy max-weight merge, min pairwise
# count=2, cluster size capped at 6 so it can't transitively chain into one
# blob). This is a SNAPSHOT from your 82.3% run. Re-derive after every future
# retrain — see the function below, just point it at your new
# confusion_matrix.csv.
# ============================================================================

CONFUSION_CLUSTERS = [
    {'A10', 'F15', 'F16', 'F18', 'F35', 'JH7'},
    {'A400M', 'B52', 'C17', 'C5', 'KC135', 'Y20'},
    {'AH64', 'J35', 'Ka52', 'TB001', 'Z10', 'Z19'},
    {'AKINCI', 'H6', 'J36', 'Tejas', 'Tu95', 'WZ7'},
    {'AV8B', 'F14', 'FCK1', 'JF17', 'Mig31', 'Su24'},
    {'An22', 'EMB314', 'MQ9', 'RQ4', 'Su25', 'TB2'},
    {'B1', 'EF2000', 'F4', 'J20', 'Mig29', 'Su34'},
    {'C130', 'CL415', 'E7', 'KJ600', 'P3', 'U2'},
    {'Il76', 'J10', 'JAS39', 'Mirage2000', 'Rafale', 'Tu22M'},
    {'B2', 'Mi24', 'Mi26', 'Mi28', 'YF23'},
    {'Be200', 'C1', 'C2', 'C390', 'XB70'},
    {'CH47', 'KAAN', 'Mi8', 'UH60', 'V22'},
    {'F22', 'Su47', 'Su57'},
    {'An124', 'An72'},
]

HARD_PAIR_CLASSES = set().union(*CONFUSION_CLUSTERS)


def derive_clusters_from_confusion_csv(csv_path, min_count=2, max_cluster_size=6):
    """Rebuild CONFUSION_CLUSTERS from a confusion_matrix.csv (rows=true,
    cols=predicted, same shape as evaluate_and_report() below produces).

    Algorithm: treat every class pair (i, j) with max(cm[i,j], cm[j,i]) >=
    min_count as a candidate edge, weight = that count. Process edges
    strongest-first (Kruskal-style) and union two classes only if the merged
    cluster would stay <= max_cluster_size. This avoids the failure mode of
    plain single-linkage clustering, which chains transitively (A confuses
    with B, B with C => A,B,C merged even if A and C never co-occur) and
    blows up into one giant, useless cluster almost immediately on real
    confusion matrices.

    Usage after any future eval run:
        clusters = derive_clusters_from_confusion_csv('confusion_matrix.csv')
        # paste the printed result over CONFUSION_CLUSTERS above
    """
    cm = pd.read_csv(csv_path, index_col=0)
    classes = cm.index.tolist()

    edges = []
    for i in classes:
        for j in classes:
            if i < j:
                w = max(cm.loc[i, j], cm.loc[j, i])
                if w >= min_count:
                    edges.append((w, i, j))
    edges.sort(key=lambda e: -e[0])

    parent = {c: c for c in classes}
    size = {c: 1 for c in classes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for w, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj and size[ri] + size[rj] <= max_cluster_size:
            parent[ri] = rj
            size[rj] += size[ri]

    groups = defaultdict(list)
    for c in classes:
        groups[find(c)].append(c)
    clusters = [sorted(g) for g in groups.values() if len(g) > 1]
    clusters.sort(key=len, reverse=True)

    print(f"Derived {len(clusters)} clusters covering "
          f"{sum(len(c) for c in clusters)}/{len(classes)} classes:")
    for c in clusters:
        print(f"    {{{', '.join(repr(x) for x in c)}}},")
    return clusters


# ============================================================================
# CONFUSION CLUSTER LOSS — unchanged logic. Now actually fires regularly
# because ClusterAwareBatchSampler (below) deliberately puts same-cluster
# pairs in the same batch instead of relying on chance with batch_size=4.
# ============================================================================

class ConfusionClusterLoss(nn.Module):
    def __init__(self, cluster_index_map, margin=0.5):
        super().__init__()
        self.cluster_map = cluster_index_map
        self.margin = margin

    def forward(self, pooled_features, targets):
        feats = F.normalize(pooled_features, dim=1)
        B = feats.size(0)
        losses = []
        for i in range(B):
            ci = self.cluster_map.get(targets[i].item(), -1)
            if ci < 0:
                continue
            for j in range(i + 1, B):
                cj = self.cluster_map.get(targets[j].item(), -1)
                if cj == ci and targets[i] != targets[j]:
                    dist = (feats[i] - feats[j]).pow(2).sum().sqrt()
                    losses.append(F.relu(self.margin - dist))
        if not losses:
            return torch.tensor(0.0, device=pooled_features.device)
        return torch.stack(losses).mean()


# ============================================================================
# EFFECTIVE-NUMBER-OF-SAMPLES CLASS WEIGHTS (Cui et al., 2019) — unchanged.
# The math was always fine; it just wasn't reaching most of the loss (see
# change #1 above).
# ============================================================================

def effective_num_class_weights(class_counts, num_classes, beta=0.999):
    counts = np.array([class_counts.get(i, 1) for i in range(num_classes)], dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.clip(effective_num, 1e-8, None)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# SHARED IMAGE TRANSFORM — extracted so JetAircraftDataset (train, xlsx,
# folder-per-class) and FlatTestDataset (flat dir, txt, no folders) apply
# IDENTICAL preprocessing. Divergent transform code between train/test
# datasets is a classic silent source of "why is my eval worse than it
# should be" bugs, so there is exactly one implementation now.
# ============================================================================

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_and_transform_image(image_path, bbox, is_train):
    img = Image.open(image_path).convert('RGB')
    w, h = img.size

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        px = int((x1 - x0) * 0.05)
        py = int((y1 - y0) * 0.05)
        img = img.crop((max(0, x0 - px), max(0, y0 - py),
                         min(w, x1 + px), min(h, y1 + py)))

    img = img.resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0

    if is_train:
        top = np.random.randint(0, 33)
        left = np.random.randint(0, 33)
        arr = arr[top:top + 224, left:left + 224]
        if np.random.rand() > 0.5:
            arr = np.fliplr(arr).copy()
        if np.random.rand() < 0.5:
            arr = (arr * np.random.uniform(0.7, 1.3)).clip(0, 1)
        if np.random.rand() < 0.25:
            arr = np.power(arr.clip(1e-6, 1), np.random.uniform(0.35, 0.7)).astype(np.float32)
        if np.random.rand() < 0.25:
            mv = arr.mean(axis=(0, 1))
            for _ in range(np.random.randint(1, 3)):
                ph = np.random.randint(28, 75); pw = np.random.randint(28, 75)
                py_ = np.random.randint(0, 224 - ph); px_ = np.random.randint(0, 224 - pw)
                arr[py_:py_ + ph, px_:px_ + pw] = mv
    else:
        arr = cv2.resize(arr, (224, 224))

    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    return (t - IMAGENET_MEAN) / IMAGENET_STD


# ============================================================================
# DATASET — train / internal-val. Same folder+xlsx layout and largest-area
# same-class-name matching as v1. Only change: per-class skip tracking so
# you can see whether a specific class (e.g. TB2) has an abnormal rate of
# "no matching-class bbox, fell back to full image."
# ============================================================================

class JetAircraftDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', images_subdir='org',
                 val_fraction=0.15, seed=42, class_to_idx=None):
        root = Path(root_dir)
        self.is_train = (split == 'train')

        split_dir = root / split / images_subdir
        train_dir = root / 'train' / images_subdir
        has_own_split_dir = split_dir.exists()
        source_dir = split_dir if has_own_split_dir else train_dir
        if not source_dir.exists():
            raise FileNotFoundError(f"Cannot find images under {source_dir}")

        self.classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        self.class_to_idx = class_to_idx or {c: i for i, c in enumerate(self.classes)}
        self.hard_class_indices = {
            self.class_to_idx[c] for c in HARD_PAIR_CLASSES if c in self.class_to_idx
        }

        all_samples = self._build_index(source_dir)

        if has_own_split_dir:
            self.samples = all_samples
        else:
            rng = np.random.RandomState(seed)
            by_class = {}
            for s in all_samples:
                by_class.setdefault(s['label'], []).append(s)
            train_s, val_s = [], []
            for label, items in by_class.items():
                idxs = rng.permutation(len(items))
                n_val = max(1, int(len(items) * val_fraction)) if len(items) > 3 else 1
                val_idx = set(idxs[:n_val].tolist())
                for i, s in enumerate(items):
                    (val_s if i in val_idx else train_s).append(s)
            self.samples = train_s if split == 'train' else val_s

        self.class_counts = {}
        for s in self.samples:
            self.class_counts[s['label']] = self.class_counts.get(s['label'], 0) + 1

        print(f"{split.upper()}: {len(self.samples)} images, {len(self.classes)} classes")

    def _build_index(self, source_dir):
        cache_path = source_dir / '.annot_cache.pkl'
        fingerprint = hashlib.md5(
            ''.join(sorted(p.name for p in source_dir.rglob('*.jpg'))).encode()
        ).hexdigest()

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                if cached.get('fingerprint') == fingerprint:
                    print(f"  Loaded annotation cache: {cache_path}")
                    return cached['samples']
            except Exception:
                pass

        print(f"  Building annotation index for {source_dir} (one-time cost)...")
        samples = []
        skipped_no_match_by_class = Counter()   # CHANGED: per-class, not just a global int
        cls_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
        n_done = 0
        for cls_dir in cls_dirs:
            cls_name = cls_dir.name
            if cls_name not in self.class_to_idx:
                continue
            label = self.class_to_idx[cls_name]
            for img_path in cls_dir.glob('*.jpg'):
                ann_path = img_path.with_suffix('.xlsx')
                bbox = None
                if ann_path.exists():
                    bbox = self._match_bbox(ann_path, cls_name)
                    if bbox is None:
                        skipped_no_match_by_class[cls_name] += 1
                samples.append({
                    'image_path': img_path,
                    'bbox': bbox,
                    'label': label,
                    'is_hard': cls_name in HARD_PAIR_CLASSES,
                })
                n_done += 1
                if n_done % 2000 == 0:
                    print(f"    ...{n_done} images indexed")

        total_skipped = sum(skipped_no_match_by_class.values())
        print(f"  Index built: {len(samples)} images "
              f"({total_skipped} had no matching-class bbox, using full image)")
        if skipped_no_match_by_class:
            print("  Per-class bbox-match-failure counts (check for classes with an "
                  "abnormally high rate here — a class relying heavily on full-image "
                  "fallback is a likely source of noisy/uninformative crops):")
            # count relative to how many images that class actually has
            cls_totals = Counter(s['image_path'].parent.name for s in samples
                                  if s['image_path'].parent.name in skipped_no_match_by_class
                                  or True)
            for cls_name, n_skip in skipped_no_match_by_class.most_common():
                cls_total = sum(1 for s in samples if s['image_path'].parent.name == cls_name)
                pct = 100 * n_skip / max(cls_total, 1)
                print(f"    {cls_name:14s} {n_skip:4d}/{cls_total:4d} ({pct:5.1f}%) full-image fallback")

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({'fingerprint': fingerprint, 'samples': samples}, f)
        except Exception as e:
            print(f"  (Could not write annotation cache: {e})")

        return samples

    def _match_bbox(self, ann_path, target_class):
        try:
            df = pd.read_excel(ann_path)
        except Exception:
            return None
        if 'class' not in df.columns:
            return None
        rows = df[df['class'].astype(str).str.strip() == target_class]
        if rows.empty:
            return None
        best = None
        best_area = -1
        for _, r in rows.iterrows():
            try:
                x0, y0, x1, y1 = int(r['xmin']), int(r['ymin']), int(r['xmax']), int(r['ymax'])
            except Exception:
                continue
            if x1 <= x0 or y1 <= y0:
                continue
            area = (x1 - x0) * (y1 - y0)
            if area > best_area:
                best_area = area
                best = (x0, y0, x1, y1)
        return best

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        t = load_and_transform_image(s['image_path'], s['bbox'], self.is_train)
        return t, s['label']


# ============================================================================
# FLAT TEST-FGVC DATASET (new) — for a directory like
#   test-fgvc/test/<hash>.jpg
#   test-fgvc/test/<hash>.txt      (filename, width, height, class, xmin,
#                                    ymin, xmax, ymax — same columns as your
#                                    xlsx, but flat, no class subfolders, and
#                                    a plain-text file per image instead of
#                                    xlsx)
#
# No folder name to tell us "this is the labeled subject" here, so instead:
# every object row whose class is in the training vocabulary becomes its
# own (crop, label) sample. A row for a class NOT in your 88-class
# vocabulary (e.g. a civilian AirbusA380 sharing the frame with a Mig29) is
# simply dropped. An image with two valid target-class boxes yields two
# separate evaluation samples. An image with zero valid boxes is skipped.
# Robust to tab / comma / whitespace-delimited txt files and an optional
# header row.
# ============================================================================

class FlatTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, class_to_idx, img_exts=('.jpg', '.jpeg', '.png')):
        self.root = Path(root_dir)
        self.class_to_idx = class_to_idx
        self.classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
        if not self.root.exists():
            raise FileNotFoundError(f"Cannot find test-fgvc directory: {self.root}")

        # CHANGED: case/hyphen/space-insensitive fallback lookup. Real-world
        # annotation files routinely spell classes slightly differently from
        # your folder names ("Mig-29" vs "Mig29", "F 16" vs "F16", trailing
        # whitespace, etc). Exact match is tried first; this is the fallback
        # that stops those from silently vanishing into "out of vocabulary."
        self._norm_to_label = {}
        for c, idx in class_to_idx.items():
            self._norm_to_label.setdefault(self._normalize(c), idx)

        self.samples = self._build_index(img_exts)
        print(f"FLAT TEST-FGVC: {len(self.samples)} evaluation samples "
              f"from {self.root}")

        if len(self.samples) < 10:
            print("  *** WARNING: very few (or zero) usable samples. This almost "
                  "always means the 'class' strings in your .txt files don't match "
                  "your training folder names (case, hyphens, spacing, etc). "
                  "See the 'unmatched class strings seen' list printed above — "
                  "if you see near-misses of real class names there, that's the bug. ***")

    @staticmethod
    def _normalize(s):
        return re.sub(r'[^a-z0-9]', '', s.lower())

    def _resolve_class(self, raw_cls):
        """Exact match first, then normalized (case/punctuation-insensitive)
        fallback. Returns the class_to_idx key to use, or None if truly
        unmatched (e.g. a genuine out-of-vocabulary distractor)."""
        if raw_cls in self.class_to_idx:
            return raw_cls
        norm = self._normalize(raw_cls)
        idx = self._norm_to_label.get(norm)
        if idx is not None:
            return self.classes[idx]
        return None

    # Bump this if the parsing/matching logic changes again in the future —
    # it invalidates any stale .flat_test_cache.pkl automatically instead of
    # requiring you to remember to delete it by hand.
    _CACHE_VERSION = 2

    def _build_index(self, img_exts):
        cache_path = self.root / '.flat_test_cache.pkl'
        img_paths = sorted(p for p in self.root.iterdir() if p.suffix.lower() in img_exts)
        fingerprint = hashlib.md5(
            (str(self._CACHE_VERSION) + ''.join(p.name for p in img_paths)).encode()
        ).hexdigest()

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                if cached.get('fingerprint') == fingerprint:
                    print(f"  Loaded flat-test annotation cache: {cache_path}")
                    return cached['samples']
            except Exception:
                pass

        samples = []
        skipped_no_txt = 0
        skipped_no_valid_class = 0
        skipped_out_of_vocab_rows = 0
        unmatched_raw_classes = Counter()   # NEW: diagnostic — what did we actually see?
        rescued_via_normalize = Counter()   # NEW: matches that only worked after normalization

        for img_path in img_paths:
            txt_path = img_path.with_suffix('.txt')
            if not txt_path.exists():
                skipped_no_txt += 1
                continue
            objs = self._parse_annotation_txt(txt_path)
            valid = []
            for o in objs:
                resolved = self._resolve_class(o['class'])
                if resolved is None:
                    unmatched_raw_classes[o['class']] += 1
                    continue
                if resolved != o['class']:
                    rescued_via_normalize[f"{o['class']!r} -> {resolved!r}"] += 1
                valid.append({'class': resolved, 'bbox': o['bbox']})
            skipped_out_of_vocab_rows += len(objs) - len(valid)
            if not valid:
                skipped_no_valid_class += 1
                continue
            for o in valid:
                samples.append({
                    'image_path': img_path,
                    'bbox': o['bbox'],
                    'label': self.class_to_idx[o['class']],
                })

        print(f"  Flat test index: {len(img_paths)} images -> {len(samples)} samples")
        print(f"    skipped (no .txt found): {skipped_no_txt}")
        print(f"    skipped (only out-of-vocab classes annotated): {skipped_no_valid_class}")
        print(f"    out-of-vocab rows dropped (kept sibling in-vocab rows if any): "
              f"{skipped_out_of_vocab_rows}")
        if rescued_via_normalize:
            print(f"    rescued via case/punctuation-insensitive match ({sum(rescued_via_normalize.values())} rows) — "
                  f"consider fixing these at the source so exact-match stays reliable:")
            for k, n in rescued_via_normalize.most_common(15):
                print(f"      {k}  x{n}")
        if unmatched_raw_classes:
            print(f"    unmatched class strings seen ({len(unmatched_raw_classes)} distinct, "
                  f"{sum(unmatched_raw_classes.values())} rows total) — top 20 by frequency:")
            for cls_str, n in unmatched_raw_classes.most_common(20):
                print(f"      {cls_str!r}  x{n}")

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({'fingerprint': fingerprint, 'samples': samples}, f)
        except Exception as e:
            print(f"  (Could not write flat-test cache: {e})")

        return samples

    @staticmethod
    def _parse_annotation_txt(txt_path):
        """Parse filename,width,height,class,xmin,ymin,xmax,ymax rows.
        Handles tab / comma / arbitrary-whitespace delimiters and an
        optional header row (detected by the first field literally being
        'filename')."""
        rows = []
        try:
            text = Path(txt_path).read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return rows

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                fields = [f.strip() for f in line.split(',')]
            else:
                fields = line.split()
            if len(fields) < 8:
                continue
            fname, width, height, cls, xmin, ymin, xmax, ymax = fields[:8]
            if fname.lower() == 'filename':
                continue
            try:
                xmin, ymin, xmax, ymax = (int(float(xmin)), int(float(ymin)),
                                           int(float(xmax)), int(float(ymax)))
            except ValueError:
                continue
            if xmax <= xmin or ymax <= ymin:
                continue
            rows.append({'class': cls, 'bbox': (xmin, ymin, xmax, ymax)})
        return rows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        t = load_and_transform_image(s['image_path'], s['bbox'], is_train=False)
        return t, s['label']


# ============================================================================
# CLUSTER-AWARE / CLASS-BALANCED BATCH SAMPLER (new)
#
# Two problems this solves at once:
#   1. ConfusionClusterLoss needs a same-cluster-different-class pair inside
#      one batch to produce any gradient. At batch_size=4 with plain
#      shuffling across 88 classes that's rare, so the loss term was mostly
#      dead weight. This sampler deliberately builds a large fraction of
#      batches as 2 classes x 2 samples drawn from the same cluster.
#   2. Tail classes (Z19 at ~65 images) get statistically drowned out by
#      random shuffling over an epoch. The non-cluster fallback batches draw
#      classes with probability inversely proportional to their frequency,
#      so rare classes show up far more often per epoch than their raw
#      count would suggest.
# ============================================================================

class ClusterAwareBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, clusters, class_to_idx, batch_size=4,
                 p_cluster=0.55, num_batches=None, seed=0):
        self.batch_size = batch_size
        self.p_cluster = p_cluster
        self.rng = np.random.RandomState(seed)

        self.indices_by_class = defaultdict(list)
        for i, s in enumerate(dataset.samples):
            self.indices_by_class[s['label']].append(i)

        self.cluster_members = []  # list of lists-of-label-ints, only clusters with >=2 present classes
        for clset in clusters:
            label_ids = [class_to_idx[c] for c in clset
                         if c in class_to_idx and class_to_idx[c] in self.indices_by_class]
            if len(label_ids) >= 2:
                self.cluster_members.append(label_ids)

        self.all_labels = list(self.indices_by_class.keys())
        counts = np.array([len(self.indices_by_class[l]) for l in self.all_labels], dtype=np.float64)
        inv_freq = 1.0 / np.clip(counts, 1, None)
        self.weights = inv_freq / inv_freq.sum()

        self.num_batches = num_batches or (len(dataset) // batch_size)

    def _sample_class(self, label):
        pool = self.indices_by_class[label]
        return pool[self.rng.randint(len(pool))]

    def __iter__(self):
        k = self.batch_size // 2
        for _ in range(self.num_batches):
            batch = []
            if self.cluster_members and self.rng.rand() < self.p_cluster:
                cluster = self.cluster_members[self.rng.randint(len(self.cluster_members))]
                c1, c2 = self.rng.choice(cluster, size=2, replace=False)
                batch = [self._sample_class(c1) for _ in range(k)] + \
                         [self._sample_class(c2) for _ in range(self.batch_size - k)]
            else:
                chosen = self.rng.choice(self.all_labels, size=self.batch_size, p=self.weights)
                batch = [self._sample_class(c) for c in chosen]
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# ============================================================================
# BACKBONE — ViT-B/16 (unchanged from v1)
# ============================================================================

class ViTB16Backbone(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        net = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.conv_proj = net.conv_proj
        self.class_token = net.class_token
        self.encoder = net.encoder
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(768, num_classes)

    def _encode(self, x):
        B = x.shape[0]
        x = self.conv_proj(x).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return self.encoder(x)

    def _encode_checkpointed(self, x):
        from torch.utils.checkpoint import checkpoint
        B = x.shape[0]
        x = self.conv_proj(x).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.encoder.pos_embedding
        x = self.encoder.dropout(x)
        for block in self.encoder.layers:
            x = checkpoint(block, x, use_reentrant=False)
        return self.encoder.ln(x)

    def forward(self, x):
        tokens = self._encode(x)
        cls = tokens[:, 0]
        return self.fc(self.dropout(cls))

    def forward_with_feat(self, x):
        tokens = self._encode_checkpointed(x)
        cls = tokens[:, 0]
        patches = tokens[:, 1:]
        B = x.shape[0]
        feat = patches.reshape(B, 14, 14, 768).permute(0, 3, 1, 2).contiguous()
        logits = self.fc(self.dropout(cls))
        return feat, cls, logits


# ============================================================================
# TAMRZ — attn_conv dead-gradient fix (from v1) kept as-is. NEW: zoom-crop
# feature extraction can now run WITH gradient on a schedule (run_zoom_grad),
# using the same checkpointed encoder as the main forward pass so memory
# stays bounded. Also NEW: class_weights threaded through loss_zoom /
# loss_attn so the imbalance correction reaches these branches too.
# ============================================================================

class TAMRZModule(nn.Module):
    def __init__(self, num_classes, zoom_ratios=(0.60, 0.35)):
        super().__init__()
        self.zoom_ratios = zoom_ratios
        self.attn_conv = nn.Conv2d(768, 1, kernel_size=1)
        self.attn_head = nn.Linear(768, num_classes)
        self.zoom_head = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, num_classes),
        )
        self.level_weights = nn.Parameter(torch.tensor([0.50, 0.30, 0.20]))

    def get_attn(self, feat):
        a = F.relu(self.attn_conv(feat)).squeeze(1)
        B, H, W = a.shape
        flat = a.view(B, -1)
        attn = (flat / (flat.max(1, keepdim=True)[0] + 1e-8)).view(B, H, W)
        return attn

    def _attn_pooled_logits(self, feat, attn):
        attn_norm = attn / (attn.sum(dim=(1, 2), keepdim=True) + 1e-8)
        pooled_attn = (feat * attn_norm.unsqueeze(1)).sum(dim=(2, 3))
        return self.attn_head(pooled_attn)

    def _zoom_crops(self, images, attn):
        B, C, H, W = images.shape
        attn_up = F.interpolate(
            attn.unsqueeze(1).detach(), size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(1)
        crops_list = []
        for ratio in self.zoom_ratios:
            ch, cw = int(H * ratio), int(W * ratio)
            crops = []
            for b in range(B):
                fi = attn_up[b].argmax()
                cy, cx = int(fi // W), int(fi % W)
                y1 = min(max(0, cy - ch // 2), H - ch)
                x1 = min(max(0, cx - cw // 2), W - cw)
                c = images[b:b + 1, :, y1:y1 + ch, x1:x1 + cw]
                crops.append(F.interpolate(c, (224, 224), mode='bilinear', align_corners=False))
            crops_list.append(torch.cat(crops, dim=0))
        return crops_list

    def _extract_zoom_feat(self, backbone, zc, use_grad):
        # CHANGED: use_grad lets the backbone actually learn from zoomed-in
        # crops on a fraction of steps, instead of always being detached.
        if use_grad:
            tokens = backbone._encode_checkpointed(zc)
        else:
            with torch.no_grad():
                tokens = backbone._encode(zc)
        return tokens[:, 0]

    def forward(self, backbone, feat, pooled, logits_g, images, targets=None,
                class_weights=None, run_zoom_grad=False, logit_adjustment=None):
        attn = self.get_attn(feat)

        with torch.no_grad():
            zoom_crops_list = self._zoom_crops(images, attn)

        zoom_logits = []
        for zc in zoom_crops_list:
            zp = self._extract_zoom_feat(backbone, zc, use_grad=run_zoom_grad)
            zoom_logits.append(self.zoom_head(zp))

        w = F.softmax(self.level_weights, dim=0)
        # NOTE: `fused` (the return value, used for accuracy/eval/predictions)
        # is built from the RAW zoom_logits — logit_adjustment below only
        # touches a local copy used for the loss, never this value.
        fused = w[0] * logits_g + sum(w[i + 1] * zl for i, zl in enumerate(zoom_logits))

        loss_zoom = None
        loss_attn = None
        if targets is not None:
            adj = logit_adjustment if logit_adjustment is not None else 0.0
            loss_zoom = sum(
                F.cross_entropy(zl + adj, targets, weight=class_weights, label_smoothing=0.05)
                for zl in zoom_logits
            ) / len(zoom_logits)
            w_ent = -(w * (w + 1e-8).log()).sum()
            loss_zoom = loss_zoom - 0.05 * w_ent

            attn_logits = self._attn_pooled_logits(feat, attn)
            loss_attn = F.cross_entropy(attn_logits + adj, targets, weight=class_weights,
                                         label_smoothing=0.05)

        return fused, loss_zoom, loss_attn


# ============================================================================
# SPATIAL UNCERTAINTY HEAD (unchanged)
# ============================================================================

class SpatialUncertaintyHead(nn.Module):
    def __init__(self, in_channels=768, num_classes=88, dropout_p=0.15):
        super().__init__()
        self.p = dropout_p
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, feat):
        return self.conv(feat)

    def mc_uncertainty(self, feat, T=2):
        was_training = self.training
        self.train()
        preds = [
            F.softmax(self.conv(F.dropout2d(feat.detach(), p=self.p, training=True)), dim=1)
            for _ in range(T)
        ]
        self.train(was_training)
        mean_p = torch.stack(preds).mean(0)
        return -(mean_p * (mean_p + 1e-8).log()).sum(dim=1)


# ============================================================================
# INTERVENTION GENERATOR (unchanged from v1)
# ============================================================================

class InterventionGenerator(nn.Module):
    def __init__(self, use_inpainting=False):
        super().__init__()
        self.use_inpainting = use_inpainting
        n_types = 3 if use_inpainting else 2
        self.w = nn.Parameter(torch.ones(n_types) / n_types)

    def generate(self, images, regions):
        batches, meta = [], []
        for b, regs in enumerate(regions):
            for r, reg in enumerate(regs):
                mask = self._mask(reg, images.shape[-2:]).to(images.device)
                t_idx = 0
                if self.use_inpainting:
                    batches.append(self._inpaint(images[b], mask)); meta.append((b, r, t_idx)); t_idx += 1
                batches.append(self._noise(images[b], mask)); meta.append((b, r, t_idx)); t_idx += 1
                batches.append(self._swap(images[b], mask, reg)); meta.append((b, r, t_idx))
        if not batches:
            return None, None
        return torch.stack(batches), meta

    def _mask(self, reg, shape):
        H, W = shape
        m = torch.zeros(H, W, dtype=torch.bool)
        x, y, w, h = reg['x'], reg['y'], reg['w'], reg['h']
        m[y:min(y + h, H), x:min(x + w, W)] = True
        return m

    def _inpaint(self, img, mask):
        np_img = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        np_mask = mask.cpu().numpy().astype(np.uint8) * 255
        out = cv2.inpaint(np_img, np_mask, 3, cv2.INPAINT_TELEA)
        return torch.from_numpy(out).float().div(255).permute(2, 0, 1).to(img.device)

    def _noise(self, img, mask):
        x = img.clone()
        if mask.sum() == 0:
            return x
        m3 = mask.unsqueeze(0).expand(3, -1, -1)
        x[m3] = (x[m3] + torch.randn(m3.sum().item(), device=img.device) * 0.3).clamp(0, 1)
        return x

    def _swap(self, img, mask, reg):
        x = img.clone(); H, W = img.shape[1:]
        h, w = reg['h'], reg['w']
        if h <= 0 or w <= 0 or h >= H or w >= W:
            return x
        ry = torch.randint(0, max(1, H - h), (1,)).item()
        rx = torch.randint(0, max(1, W - w), (1,)).item()
        try:
            x[:, reg['y']:reg['y'] + h, reg['x']:reg['x'] + w] = img[:, ry:ry + h, rx:rx + w]
        except Exception:
            pass
        return x


# ============================================================================
# CUGAR ViT — class-weight scope bug fixed (loss_fused, loss_sp now weighted;
# class_weights threaded into TAMRZ for loss_zoom/loss_attn too), and
# run_zoom_grad threaded through to TAMRZ.
# ============================================================================

class CUGARViT(nn.Module):
    def __init__(self, num_classes, K_max=2, cluster_loss_module=None, use_inpainting=False):
        super().__init__()
        self.backbone = ViTB16Backbone(num_classes)
        self.spatial = SpatialUncertaintyHead(768, num_classes, 0.15)
        self.tamrz = TAMRZModule(num_classes, zoom_ratios=(0.60, 0.35))
        self.intervener = InterventionGenerator(use_inpainting=use_inpainting)
        self.cluster_loss = cluster_loss_module
        self.K_max = K_max
        self.register_buffer('class_weights', torch.ones(num_classes))
        # NEW (v3): additive logit-adjustment term (Menon et al. 2021),
        # applied only inside the loss computation below — never to the
        # logits this module actually returns/predicts with.
        self.register_buffer('logit_adjustment', torch.zeros(num_classes))

    def set_class_weights(self, base_weights, hard_indices=None, hard_factor=1.3):
        w = base_weights.clone()
        if hard_indices:
            idx = list(hard_indices)
            w[idx] = w[idx] * hard_factor
        self.class_weights.copy_(w)

    def set_logit_adjustment(self, class_counts, num_classes, tau=1.0, count_floor=150):
        """tau * log(class prior), with each class's count floored at
        `count_floor` before computing the prior. The floor is the v4 fix:
        at tau=1.0 with raw counts, a 57-image class (Z19) was forced to
        overcome a ~3.0 logit margin against a 1104-image class (F16) with
        essentially no data to back that margin up, and it overcorrected —
        precision collapsed from 0.92 to 0.57 while recall barely moved.
        Flooring the count used in the adjustment (NOT the actual class_weight
        or training exposure — only the size of this term) caps how large a
        margin a tiny class is forced to fight for. count_floor=150 verified
        to cut the F16-vs-Z19 required margin by ~33% while changing F16's
        own adjustment by <0.01 (it's already far above the floor).
        tau=0 disables this entirely (v2 behavior)."""
        counts = torch.tensor([max(class_counts.get(i, 1), count_floor) for i in range(num_classes)],
                               dtype=torch.float32)
        prior = counts / counts.sum()
        self.logit_adjustment.copy_(tau * torch.log(prior))

    @staticmethod
    def _select_regions(U, image_size, K_max, q=85):
        B, H, W = U.shape
        scale = image_size // H
        regions = []
        for b in range(B):
            scores = U[b].flatten()
            tau = torch.quantile(scores, q / 100)
            idx = (U[b] > tau).nonzero(as_tuple=False)
            if len(idx) > K_max:
                fi = idx[:, 0] * W + idx[:, 1]
                _, top = torch.topk(scores[fi], K_max)
                idx = idx[top]
            regions.append([{
                'x': c[1].item() * scale, 'y': c[0].item() * scale,
                'w': scale, 'h': scale,
            } for c in idx])
        return regions

    def forward(self, x, targets=None, training=True, run_causal=True, run_zoom_grad=False):
        if not training:
            feat, pooled, logits_g = self.backbone.forward_with_feat(x)
            fused, _, _ = self.tamrz(self.backbone, feat, pooled, logits_g, x)
            return fused

        feat, pooled, logits_g = self.backbone.forward_with_feat(x)

        # NEW (v3): logit_adjustment is added to the logits fed into every
        # cross-entropy call below, and ONLY there — `fused` / `logits_g`
        # themselves stay raw, since those are what forward(training=False)
        # returns and what accuracy is computed from.
        adj = self.logit_adjustment

        loss_cls = F.cross_entropy(
            logits_g + adj, targets, weight=self.class_weights, label_smoothing=0.05
        )
        fused, loss_zoom, loss_attn = self.tamrz(
            self.backbone, feat, pooled, logits_g, x, targets,
            class_weights=self.class_weights, run_zoom_grad=run_zoom_grad,
            logit_adjustment=adj,
        )
        # CHANGED (v2): was unweighted in v1. This is the branch model.eval() /
        # accuracy / checkpoint selection actually use — most important fix.
        loss_fused = F.cross_entropy(fused + adj, targets, weight=self.class_weights,
                                      label_smoothing=0.05)

        sp_logits = self.spatial(feat)
        # CHANGED (v2): was unweighted in v1.
        loss_sp = F.cross_entropy(sp_logits.mean(dim=(2, 3)) + adj, targets,
                                   weight=self.class_weights, label_smoothing=0.05)
        U = self.spatial.mc_uncertainty(feat, T=2)

        loss_causal = torch.tensor(0.0, device=x.device)
        if run_causal:
            regions = self._select_regions(U, x.shape[-1], self.K_max)
            mean_ = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std_ = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x_px = x * std_ + mean_

            int_batch, meta = self.intervener.generate(x_px, regions)

            if int_batch is not None:
                int_norm = (int_batch.clamp(0, 1) - mean_) / std_
                with torch.no_grad():
                    logits_int = self.backbone(int_norm)
                probs_orig = F.softmax(logits_g.detach(), dim=1)
                probs_int = F.softmax(logits_int, dim=1)
                w_int = F.softmax(self.intervener.w, dim=0)
                losses = []
                for i, (b_idx, r_idx, t_idx) in enumerate(meta):
                    tgt = targets[b_idx]
                    losses.append(w_int[t_idx] * F.relu(0.3 - probs_orig[b_idx, tgt] + probs_int[i, tgt]))
                loss_causal = torch.stack(losses).mean()

        loss_cluster = torch.tensor(0.0, device=x.device)
        if self.cluster_loss is not None:
            loss_cluster = self.cluster_loss(pooled, targets)

        total = (1.0 * loss_fused
                 + 0.3 * loss_cls
                 + 0.5 * loss_zoom
                 + 0.2 * loss_attn
                 + 0.3 * loss_sp
                 + 0.4 * loss_causal
                 + 0.3 * loss_cluster)

        return {
            'loss': total, 'loss_fused': loss_fused, 'loss_cls': loss_cls,
            'loss_zoom': loss_zoom, 'loss_attn': loss_attn, 'loss_sp': loss_sp,
            'loss_causal': loss_causal, 'loss_cluster': loss_cluster, 'logits': fused,
        }


# ============================================================================
# HELPERS
# ============================================================================

def build_cluster_loss(train_ds):
    cluster_map = {}
    for cid, cls_set in enumerate(CONFUSION_CLUSTERS):
        for cls_name in cls_set:
            idx = train_ds.class_to_idx.get(cls_name)
            if idx is not None:
                cluster_map[idx] = cid
    print(f"Cluster loss: {len(cluster_map)} classes across {len(CONFUSION_CLUSTERS)} clusters")
    return ConfusionClusterLoss(cluster_map, margin=0.5)


def tta_predict(model, images, device):
    model.eval()
    mean_ = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_ = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x_px = (images * std_ + mean_).clamp(0, 1)
    with torch.no_grad():
        p1 = F.softmax(model(images, training=False), dim=1)
        p2 = F.softmax(model((torch.flip(x_px, [3]) - mean_) / std_, training=False), dim=1)
        xc = F.interpolate(x_px, (256, 256), mode='bilinear', align_corners=False)[:, :, 16:240, 16:240]
        p3 = F.softmax(model((xc - mean_) / std_, training=False), dim=1)
    return (p1 + p2 + p3) / 3.0


# ============================================================================
# EVALUATE + REPORT (new) — runs TTA inference over any dataset, builds a
# confusion matrix, and writes the same 4 CSVs: per_class_metrics.csv,
# confusion_matrix.csv, top_confused_pairs.csv (exhaustive), and
# top_confusions_per_class.csv (grouped "true_class / PredClass - count").
# ============================================================================

def evaluate_and_report(model, dataset, device, class_names, out_dir,
                         batch_size=8, use_tta=True, top_k_per_class=5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)

    if len(dataset) == 0:
        # CHANGED: don't crash — this is exactly what happened when the flat
        # test-fgvc set indexed 0 samples (a class-name mismatch). Write
        # empty-but-correctly-shaped CSVs and return cleanly so training
        # keeps going; the printed warning in FlatTestDataset tells you why.
        print("  evaluate_and_report: dataset has 0 samples, skipping "
              "evaluation and writing empty report CSVs.")
        empty_pcm = pd.DataFrame(columns=['class', 'support', 'precision', 'recall', 'f1'])
        empty_pairs = pd.DataFrame(columns=['true_class', 'predicted_class', 'count'])
        empty_pcm.to_csv(out_dir / 'per_class_metrics.csv', index=False)
        empty_pairs.to_csv(out_dir / 'top_confused_pairs.csv', index=False)
        pd.DataFrame(np.zeros((num_classes, num_classes), dtype=int),
                     index=class_names, columns=class_names).to_csv(out_dir / 'confusion_matrix.csv')
        with open(out_dir / 'top_confusions_per_class.csv', 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=['true_class', 'misspredicted_class']).writeheader()
        return {'overall_acc': 0.0, 'confusion_matrix': None,
                'per_class_metrics': empty_pcm, 'top_confused_pairs': empty_pairs}

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if use_tta:
                probs = tta_predict(model, images, device)
                preds = probs.argmax(1)
            else:
                preds = model(images, training=False).argmax(1)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                cm[t, p] += 1

    # ---- confusion_matrix.csv ----
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(out_dir / 'confusion_matrix.csv')

    # ---- per_class_metrics.csv ----
    rows = []
    for i, cls in enumerate(class_names):
        support = cm[i, :].sum()
        tp = cm[i, i]
        pred_total = cm[:, i].sum()
        precision = tp / pred_total if pred_total > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append({'class': cls, 'support': int(support), 'precision': round(precision, 4),
                      'recall': round(recall, 4), 'f1': round(f1, 4)})
    pcm_df = pd.DataFrame(rows)
    pcm_df.to_csv(out_dir / 'per_class_metrics.csv', index=False)

    # ---- top_confused_pairs.csv (exhaustive, all nonzero off-diagonal) ----
    pair_rows = []
    for i, true_cls in enumerate(class_names):
        for j, pred_cls in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                pair_rows.append({'true_class': true_cls, 'predicted_class': pred_cls,
                                   'count': int(cm[i, j])})
    # CHANGED: pd.DataFrame([]) has no columns to sort by — construct with
    # explicit columns so an all-correct (or all-empty) matrix can't crash this.
    pairs_df = pd.DataFrame(pair_rows, columns=['true_class', 'predicted_class', 'count'])
    pairs_df = pairs_df.sort_values('count', ascending=False)
    pairs_df.to_csv(out_dir / 'top_confused_pairs.csv', index=False)

    # ---- top_confusions_per_class.csv (grouped, requested format) ----
    grouped_rows = []
    if len(pairs_df) > 0:
        totals = pairs_df.groupby('true_class')['count'].sum().sort_values(ascending=False)
        g = pairs_df.groupby('true_class')
        for true_cls in totals.index:
            sub = g.get_group(true_cls).sort_values('count', ascending=False).head(top_k_per_class)
            first = True
            for _, r in sub.iterrows():
                grouped_rows.append({
                    'true_class': true_cls if first else '',
                    'misspredicted_class': f"{r['predicted_class']} - {int(r['count'])}",
                })
                first = False
    with open(out_dir / 'top_confusions_per_class.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['true_class', 'misspredicted_class'])
        w.writeheader()
        w.writerows(grouped_rows)

    overall_acc = 100 * np.trace(cm) / max(cm.sum(), 1)
    print(f"  evaluate_and_report: {cm.sum()} samples, overall acc = {overall_acc:.2f}%, "
          f"reports written to {out_dir}")
    return {'overall_acc': overall_acc, 'confusion_matrix': cm,
            'per_class_metrics': pcm_df, 'top_confused_pairs': pairs_df}


# ============================================================================
# DEBUG UTILITY (new) — save a contact-sheet JPG of the exact crops the
# model is fed for N samples of a given class. Use this BEFORE assuming a
# low-recall class (e.g. TB2) needs a modeling fix — if the crops are mostly
# full uncropped scenes or clearly wrong content, it's a data/annotation fix
# instead.
# ============================================================================

def dump_class_samples(dataset, class_name, out_path, n=16, grid_cols=4):
    class_to_idx = dataset.class_to_idx
    if class_name not in class_to_idx:
        print(f"'{class_name}' not found in dataset vocabulary.")
        return
    target_label = class_to_idx[class_name]
    matches = [s for s in dataset.samples if s['label'] == target_label][:n]
    if not matches:
        print(f"No samples found for class '{class_name}'.")
        return

    thumb = 224
    rows = (len(matches) + grid_cols - 1) // grid_cols
    sheet = Image.new('RGB', (grid_cols * thumb, rows * thumb), (30, 30, 30))

    for i, s in enumerate(matches):
        try:
            img = Image.open(s['image_path']).convert('RGB')
            w, h = img.size
            if s['bbox'] is not None:
                x0, y0, x1, y1 = s['bbox']
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                img = img.crop((x0, y0, x1, y1))
            else:
                img = img.copy()  # full-image fallback case — this is the interesting one to see
            img = img.resize((thumb, thumb))
        except Exception as e:
            img = Image.new('RGB', (thumb, thumb), (200, 0, 0))
        r, c = divmod(i, grid_cols)
        sheet.paste(img, (c * thumb, r * thumb))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    n_full_fallback = sum(1 for s in matches if s['bbox'] is None)
    print(f"Saved {len(matches)} '{class_name}' crops to {out_path} "
          f"({n_full_fallback}/{len(matches)} used full-image fallback, no bbox match)")


# ============================================================================
# TRAINING
# ============================================================================

def train_cugar_jets():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    ROOT = r'C:\Users\Amma.DESKTOP-4K4SV7F\Documents\Hemanth\fine-grain'
    TEST_FGVC_ROOT = r'C:\Users\Amma.DESKTOP-4K4SV7F\Documents\Hemanth\fine-grain\test-fgvc\test'
    REPORTS_DIR = r'C:\Users\Amma.DESKTOP-4K4SV7F\Documents\Hemanth\fine-grain\reports'

    train_ds = JetAircraftDataset(ROOT, 'train')
    val_ds = JetAircraftDataset(ROOT, 'test', class_to_idx=train_ds.class_to_idx)
    test_fgvc_ds = FlatTestDataset(TEST_FGVC_ROOT, class_to_idx=train_ds.class_to_idx)

    NUM_CLASSES = len(train_ds.classes)

    BATCH_SIZE = 4
    GRAD_ACCUM = 4
    CAUSAL_EVERY_N = 4       # unchanged from v1
    ZOOM_GRAD_EVERY_N = 4    # unchanged from v2
    LOGIT_ADJUST_TAU = 1.0   # unchanged from v3
    LOGIT_ADJUST_FLOOR = 150 # NEW (v4): caps the adjustment penalty for very
                             # small classes (e.g. Z19 at 57 images) — see
                             # changelog. Raise if any class is still
                             # overcorrecting next run, lower for a stronger
                             # correction on mid-size classes.

    train_sampler = ClusterAwareBatchSampler(
        train_ds, CONFUSION_CLUSTERS, train_ds.class_to_idx,
        batch_size=BATCH_SIZE, p_cluster=0.55,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=0, pin_memory=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=False
    )

    cluster_loss_fn = build_cluster_loss(train_ds)
    model = CUGARViT(num_classes=NUM_CLASSES, K_max=2,
                      cluster_loss_module=cluster_loss_fn, use_inpainting=False).to(device)

    base_weights = effective_num_class_weights(train_ds.class_counts, NUM_CLASSES, beta=0.999)
    model.set_class_weights(base_weights, hard_indices=train_ds.hard_class_indices, hard_factor=1.3)
    # NEW (v3): complements class_weights rather than replacing it — see
    # changelog. LOGIT_ADJUST_TAU=0.0 reproduces exact v2 behavior if you
    # want to A/B it.
    model.set_logit_adjustment(train_ds.class_counts, NUM_CLASSES, tau=LOGIT_ADJUST_TAU,
                                count_floor=LOGIT_ADJUST_FLOOR)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.conv_proj.parameters(), 'lr': 5e-7, 'name': 'patch_embed'},
        {'params': model.backbone.encoder.parameters(), 'lr': 5e-7, 'name': 'encoder'},
        {'params': model.backbone.fc.parameters(), 'lr': 1e-4, 'name': 'cls_head'},
        {'params': model.spatial.parameters(), 'lr': 1e-4, 'name': 'spatial'},
        {'params': model.tamrz.parameters(), 'lr': 1e-4, 'name': 'tamrz'},
        {'params': model.intervener.parameters(), 'lr': 5e-5, 'name': 'intervener'},
    ], weight_decay=0.05)

    TOTAL_EPOCHS = 50
    WARMUP_EPOCHS = 10
    TARGET_ENCODER_LR = 1e-5
    TARGET_HEADS_LR = 2e-4

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS, eta_min=1e-7
    )

    best_val_acc = 0.0

    print("=" * 65)
    print(f"CUGAR ViT-B/16 v2 — {NUM_CLASSES} classes, batch=4, accum=4, "
          f"causal every {CAUSAL_EVERY_N} steps, zoom-grad every {ZOOM_GRAD_EVERY_N} steps")
    print("=" * 65 + "\n")

    for epoch in range(TOTAL_EPOCHS):
        if epoch < WARMUP_EPOCHS:
            t = (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                name = pg.get('name', '')
                if name in ('patch_embed', 'encoder'):
                    pg['lr'] = 5e-7 + (TARGET_ENCODER_LR - 5e-7) * t
                elif name in ('cls_head', 'spatial', 'tamrz', 'intervener'):
                    pg['lr'] = pg['lr'] * (0.5 + 0.5 * t)

        model.train()
        running = dict(loss=0, fused=0, zoom=0, attn=0, sp=0, causal=0, cluster=0)
        optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            run_causal = (batch_idx % CAUSAL_EVERY_N == 0)
            run_zoom_grad = (batch_idx % ZOOM_GRAD_EVERY_N == 1)  # offset from causal so they don't always coincide

            out = model(images, labels, training=True, run_causal=run_causal,
                        run_zoom_grad=run_zoom_grad)
            (out['loss'] / GRAD_ACCUM).backward()

            running['loss'] += out['loss'].item()
            running['fused'] += out['loss_fused'].item()
            running['zoom'] += out['loss_zoom'].item()
            running['attn'] += out['loss_attn'].item()
            running['sp'] += out['loss_sp'].item()
            running['causal'] += out['loss_causal'].item()
            running['cluster'] += out['loss_cluster'].item()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            if batch_idx % 100 == 0:
                enc_lr = next(pg['lr'] for pg in optimizer.param_groups if pg.get('name') == 'encoder')
                print(f"E{epoch+1} [{batch_idx:4d}/{len(train_loader)}] "
                      f"L:{out['loss'].item():.3f}  fused:{out['loss_fused'].item():.3f}  "
                      f"zm:{out['loss_zoom'].item():.3f}  attn:{out['loss_attn'].item():.3f}  "
                      f"sp:{out['loss_sp'].item():.3f}  cluster:{out['loss_cluster'].item():.3f}  "
                      f"enc_lr:{enc_lr:.2e}")

        n = len(train_loader)

        model.eval()
        correct = total = hard_correct = hard_total = correct_tta = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                preds_s = model(images, training=False).argmax(1)
                correct += (preds_s == labels).sum().item()
                total += labels.size(0)

                probs_t = tta_predict(model, images, device)
                correct_tta += (probs_t.argmax(1) == labels).sum().item()

                hm = torch.tensor(
                    [l.item() in train_ds.hard_class_indices for l in labels],
                    dtype=torch.bool, device=device
                )
                if hm.any():
                    hard_correct += (preds_s[hm] == labels[hm]).sum().item()
                    hard_total += hm.sum().item()

        val_acc = 100 * correct / total
        val_acc_tta = 100 * correct_tta / total
        hard_acc = 100 * hard_correct / hard_total if hard_total else 0.0
        zoom_w = F.softmax(model.tamrz.level_weights, dim=0)
        enc_lr = next(pg['lr'] for pg in optimizer.param_groups if pg.get('name') == 'encoder')

        print(f"\n{'='*65}")
        print(f"EPOCH {epoch+1}/{TOTAL_EPOCHS}  |  Loss: {running['loss']/n:.4f}  |  enc_lr: {enc_lr:.2e}")
        print(f"  Val  (single): {val_acc:.2f}%")
        print(f"  Val  (TTA)   : {val_acc_tta:.2f}%")
        print(f"  Hard Acc     : {hard_acc:.2f}%")
        print(f"  Zoom w: global={zoom_w[0]:.3f}  z60={zoom_w[1]:.3f}  z35={zoom_w[2]:.3f}")
        print(f"{'='*65}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc_single': val_acc,
                'val_acc_tta': val_acc_tta,
                'hard_acc': hard_acc,
                'zoom_weights': model.tamrz.level_weights.data,
                'classes': train_ds.classes,
                'class_to_idx': train_ds.class_to_idx,
            }, r'C:\Users\Amma.DESKTOP-4K4SV7F\Documents\Hemanth\fine-grain\cugar_jets_best.pth')
            print(f"✓ SAVED  single={val_acc:.2f}%  TTA={val_acc_tta:.2f}%  hard={hard_acc:.2f}%")

            # NEW: on every new best checkpoint, run the full report against
            # the real flat test-fgvc set and refresh the 4 CSVs. Wrapped in
            # try/except on purpose — a reporting bug should never be able
            # to throw away hours of training. The checkpoint above is
            # already saved by this point regardless of what happens here.
            try:
                report = evaluate_and_report(
                    model, test_fgvc_ds, device, train_ds.classes,
                    out_dir=Path(REPORTS_DIR) / f'epoch_{epoch+1}',
                )
                print(f"  test-fgvc overall acc: {report['overall_acc']:.2f}%\n")
            except Exception as e:
                print(f"  *** evaluate_and_report FAILED (training continues): {e} ***\n")

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

    print(f"\nBest Val Acc: {best_val_acc:.2f}%")


# ============================================================================
# STANDALONE EVAL ENTRY POINT — regenerate the 4 CSVs from a saved
# checkpoint against the flat test-fgvc set without retraining. Useful after
# any code/data change to just re-check where errors are now.
# ============================================================================

def evaluate_checkpoint_on_testfgvc(checkpoint_path, test_fgvc_root, out_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_idx = ckpt['class_to_idx']
    classes = ckpt['classes']

    model = CUGARViT(num_classes=len(classes)).to(device)
    try:
        model.load_state_dict(ckpt['model_state_dict'])
    except RuntimeError as e:
        # Likely an older checkpoint (e.g. v2, no logit_adjustment buffer) —
        # load what matches instead of crashing.
        print(f"  *** strict load failed ({e}); retrying with strict=False. "
              f"This is expected if this checkpoint predates a buffer this "
              f"script's CUGARViT now has (e.g. an older version). ***")
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  missing keys: {missing}\n  unexpected keys: {unexpected}")
    model.eval()

    test_ds = FlatTestDataset(test_fgvc_root, class_to_idx=class_to_idx)
    report = evaluate_and_report(model, test_ds, device, classes, out_dir=out_dir)
    print(f"Done. Overall acc on test-fgvc: {report['overall_acc']:.2f}%")
    return report


if __name__ == '__main__':
    train_cugar_jets()

    # To regenerate reports later from the saved checkpoint without
    # retraining, comment out train_cugar_jets() above and use instead:
    #
    # evaluate_checkpoint_on_testfgvc(
    #     checkpoint_path='cugar_jets_best.pth',
    #     test_fgvc_root=r'C:\Users\Amma.DESKTOP-4K4SV7F\Documents\Hemanth\fine-grain\test-fgvc\test',
    #     out_dir=r'C:\Users\Amma.DESKTOP-4K4SV7F\Documents\Hemanth\fine-grain\reports\manual_eval',
    # )