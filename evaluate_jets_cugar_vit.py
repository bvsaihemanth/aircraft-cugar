# evaluate_cugar_full_metrics.py
# Standalone evaluation script — loads a saved CUGARViT checkpoint and
# reports every metric asked for, against a YOLO-format flat test set.
#
# ANNOTATION FORMAT (this is different from the last test-fgvc set!)
# -----------------------------------------------------------------------------
# One .txt per image, one line per object:
#     class_id  x_center  y_center  width  height      (all normalized 0-1)
# e.g. "48 0.500000 0.500000 1.000000 1.000000" = class 48, full-image box.
# No filename/width/height/class-name columns like your other test set had —
# the box is normalized, so actual image dimensions are read from the image
# itself, and the label is a numeric ID, not a string name.
#
# CLASS-ID MAPPING — this is the part I can't verify from here and is the
# single biggest correctness risk in this script. By default it assumes
# class_id N corresponds to checkpoint['classes'][N], i.e. the same
# alphabetically-sorted order your training folders were indexed in. If your
# YOLO labels came from a different tool/export, that assumption could be
# wrong and every metric below would be silently meaningless. To avoid
# guessing blind, the script also looks for (in the test folder or its
# parent) a classes.txt / obj.names / data.yaml and uses that instead if
# found. EITHER WAY, check the "class-id mapping sample" printed at startup
# before trusting the numbers.
#
# Standalone by design (duplicates the CUGARViT architecture rather than
# importing your training script) so it runs regardless of what you've
# named/moved that file to.

import os, re, csv, json
from pathlib import Path
from collections import defaultdict, Counter

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
import pandas as pd
import numpy as np
import cv2
from PIL import Image

# ============================================================================
# CONFUSION_CLUSTERS — same as train_jets_cugar_vit_v2.py. Needed here for
# hard-class accuracy and the cluster / confusion-pair metrics. If you
# re-derive clusters after retraining, update this list to match.
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


# ============================================================================
# MODEL — identical to train_jets_cugar_vit_v2.py. Needs to match exactly
# (including intervener/spatial, unused at inference) so state_dict loads.
# ============================================================================

class ViTB16Backbone(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        net = models.vit_b_16(weights=None)  # weights loaded from checkpoint, not ImageNet
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

    def forward(self, x):
        tokens = self._encode(x)
        return self.fc(self.dropout(tokens[:, 0]))

    def forward_with_feat(self, x):
        tokens = self._encode(x)
        cls = tokens[:, 0]
        patches = tokens[:, 1:]
        B = x.shape[0]
        feat = patches.reshape(B, 14, 14, 768).permute(0, 3, 1, 2).contiguous()
        logits = self.fc(self.dropout(cls))
        return feat, cls, logits


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
        return (flat / (flat.max(1, keepdim=True)[0] + 1e-8)).view(B, H, W)

    def _zoom_crops(self, images, attn):
        B, C, H, W = images.shape
        attn_up = F.interpolate(attn.unsqueeze(1), size=(H, W), mode='bilinear',
                                 align_corners=False).squeeze(1)
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

    def forward(self, backbone, feat, pooled, logits_g, images):
        attn = self.get_attn(feat)
        zoom_crops_list = self._zoom_crops(images, attn)
        zoom_logits = [self.zoom_head(backbone._encode(zc)[:, 0]) for zc in zoom_crops_list]
        w = F.softmax(self.level_weights, dim=0)
        fused = w[0] * logits_g + sum(w[i + 1] * zl for i, zl in enumerate(zoom_logits))
        return fused


class SpatialUncertaintyHead(nn.Module):
    def __init__(self, in_channels=768, num_classes=88, dropout_p=0.15):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, feat):
        return self.conv(feat)


class InterventionGenerator(nn.Module):
    def __init__(self, use_inpainting=False):
        super().__init__()
        n_types = 3 if use_inpainting else 2
        self.w = nn.Parameter(torch.ones(n_types) / n_types)


class CUGARViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = ViTB16Backbone(num_classes)
        self.spatial = SpatialUncertaintyHead(768, num_classes, 0.15)
        self.tamrz = TAMRZModule(num_classes)
        self.intervener = InterventionGenerator(use_inpainting=False)
        self.register_buffer('class_weights', torch.ones(num_classes))
        # Present only so a v3-trained checkpoint's state_dict loads cleanly —
        # inert at inference (only used inside the training loss).
        self.register_buffer('logit_adjustment', torch.zeros(num_classes))

    @torch.no_grad()
    def forward(self, x):
        feat, pooled, logits_g = self.backbone.forward_with_feat(x)
        return self.tamrz(self.backbone, feat, pooled, logits_g, x)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_and_transform_image(image_path, bbox):
    """Deterministic eval transform (no augmentation). bbox is pixel xyxy or None."""
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        px = int((x1 - x0) * 0.05)
        py = int((y1 - y0) * 0.05)
        img = img.crop((max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)))
    img = img.resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = cv2.resize(arr, (224, 224))
    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def tta_predict(model, images, device):
    mean_ = IMAGENET_MEAN.to(device).unsqueeze(0)
    std_ = IMAGENET_STD.to(device).unsqueeze(0)
    x_px = (images * std_ + mean_).clamp(0, 1)
    with torch.no_grad():
        p1 = F.softmax(model(images), dim=1)
        p2 = F.softmax(model((torch.flip(x_px, [3]) - mean_) / std_), dim=1)
        xc = F.interpolate(x_px, (256, 256), mode='bilinear', align_corners=False)[:, :, 16:240, 16:240]
        p3 = F.softmax(model((xc - mean_) / std_), dim=1)
    return (p1 + p2 + p3) / 3.0


# ============================================================================
# CLASS-ID MAPPING — see the module docstring. Prefers a companion
# classes.txt / obj.names / data.yaml next to the test data; falls back to
# the checkpoint's class list.
# ============================================================================

def resolve_class_id_mapping(test_dir, checkpoint_classes):
    test_dir = Path(test_dir)
    for d in (test_dir, test_dir.parent):
        for fname in ('classes.txt', 'obj.names'):
            p = d / fname
            if p.exists():
                names = [l.strip() for l in p.read_text().splitlines() if l.strip()]
                print(f"  Using class-id mapping from {p} ({len(names)} classes)")
                return names, str(p)
    for d in (test_dir, test_dir.parent):
        p = d / 'data.yaml'
        if p.exists():
            try:
                import yaml
                data = yaml.safe_load(p.read_text())
                names = data.get('names')
                if isinstance(names, dict):
                    names = [names[i] for i in sorted(names.keys())]
                if names:
                    print(f"  Using class-id mapping from {p} ({len(names)} classes)")
                    return names, str(p)
            except Exception as e:
                print(f"  Found {p} but could not parse it ({e}), skipping")
    print(f"  No classes.txt / obj.names / data.yaml found near {test_dir} -> "
          f"falling back to checkpoint['classes'] ({len(checkpoint_classes)} classes). "
          f"*** VERIFY the sample mapping printed below is correct before trusting results. ***")
    return checkpoint_classes, "checkpoint['classes'] (fallback)"


# ============================================================================
# DATASET — flat dir, one .txt per image, YOLO-normalized boxes. Every
# object line becomes its own (crop, label) sample, same "each box is a
# sample" philosophy as the previous test-fgvc loader.
# ============================================================================

class YoloFlatTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, id_to_class, img_exts=('.jpg', '.jpeg', '.png')):
        self.root = Path(root_dir)
        self.id_to_class = id_to_class
        if not self.root.exists():
            raise FileNotFoundError(f"Cannot find test directory: {self.root}")
        self.samples = self._build_index(img_exts)
        print(f"YOLO TEST SET: {len(self.samples)} evaluation samples from {self.root}")

    @staticmethod
    def _parse_yolo_txt(txt_path):
        rows = []
        try:
            text = Path(txt_path).read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return rows
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                cls_id = int(float(fields[0]))
                xc, yc, bw, bh = (float(fields[1]), float(fields[2]),
                                   float(fields[3]), float(fields[4]))
            except ValueError:
                continue
            rows.append({'class_id': cls_id, 'xc': xc, 'yc': yc, 'w': bw, 'h': bh})
        return rows

    @staticmethod
    def _yolo_to_pixel_bbox(row, img_w, img_h):
        xc, yc = row['xc'] * img_w, row['yc'] * img_h
        bw, bh = row['w'] * img_w, row['h'] * img_h
        x0, y0 = max(0, int(round(xc - bw / 2))), max(0, int(round(yc - bh / 2)))
        x1, y1 = min(img_w, int(round(xc + bw / 2))), min(img_h, int(round(yc + bh / 2)))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _build_index(self, img_exts):
        img_paths = sorted(p for p in self.root.iterdir() if p.suffix.lower() in img_exts)
        samples = []
        skipped_no_txt = 0
        skipped_bad_id = 0
        bad_id_examples = Counter()

        for img_path in img_paths:
            txt_path = img_path.with_suffix('.txt')
            if not txt_path.exists():
                skipped_no_txt += 1
                continue
            rows = self._parse_yolo_txt(txt_path)
            if not rows:
                continue
            try:
                W, H = Image.open(img_path).size
            except Exception:
                continue
            for r in rows:
                if not (0 <= r['class_id'] < len(self.id_to_class)):
                    skipped_bad_id += 1
                    bad_id_examples[r['class_id']] += 1
                    continue
                bbox = self._yolo_to_pixel_bbox(r, W, H)
                if bbox is None:
                    continue
                samples.append({
                    'image_path': img_path,
                    'bbox': bbox,
                    'label': r['class_id'],
                })

        print(f"  {len(img_paths)} images -> {len(samples)} samples")
        print(f"    skipped (no .txt found): {skipped_no_txt}")
        if skipped_bad_id:
            print(f"    skipped (class_id out of range 0..{len(self.id_to_class)-1}): "
                  f"{skipped_bad_id} rows, ids seen: {dict(bad_id_examples.most_common(10))}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        t = load_and_transform_image(s['image_path'], s['bbox'])
        return t, s['label']


# ============================================================================
# FULL METRICS EVALUATION
# ============================================================================

def evaluate_full(model, dataset, class_names, device, out_dir, batch_size=8, top_k=5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    hard_indices = {class_to_idx[c] for c in HARD_PAIR_CLASSES if c in class_to_idx}
    cluster_of = {}  # class_idx -> cluster_id
    for cid, clset in enumerate(CONFUSION_CLUSTERS):
        for c in clset:
            if c in class_to_idx:
                cluster_of[class_to_idx[c]] = cid

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()

    all_labels, single_top1, tta_top1 = [], [], []
    single_top5_hits, tta_top5_hits = [], []
    cm_single = np.zeros((num_classes, num_classes), dtype=np.int64)
    cm_tta = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits_single = model(images)
            probs_single = F.softmax(logits_single, dim=1)
            probs_tta = tta_predict(model, images, device)

            top5_single = probs_single.topk(top_k, dim=1).indices
            top5_tta = probs_tta.topk(top_k, dim=1).indices

            pred_single = probs_single.argmax(1)
            pred_tta = probs_tta.argmax(1)

            for i in range(labels.size(0)):
                t = labels[i].item()
                ps, pt = pred_single[i].item(), pred_tta[i].item()
                all_labels.append(t)
                single_top1.append(ps)
                tta_top1.append(pt)
                single_top5_hits.append(t in top5_single[i].tolist())
                tta_top5_hits.append(t in top5_tta[i].tolist())
                cm_single[t, ps] += 1
                cm_tta[t, pt] += 1

    n = len(all_labels)
    all_labels = np.array(all_labels)
    single_top1 = np.array(single_top1)
    tta_top1 = np.array(tta_top1)

    # ---------------- headline metrics ----------------
    top1_single = 100 * (single_top1 == all_labels).mean()
    top1_tta = 100 * (tta_top1 == all_labels).mean()
    top5_single = 100 * np.mean(single_top5_hits)
    top5_tta = 100 * np.mean(tta_top5_hits)

    def hard_class_acc(preds):
        mask = np.isin(all_labels, list(hard_indices))
        if mask.sum() == 0:
            return 0.0
        return 100 * (preds[mask] == all_labels[mask]).mean()

    hard_acc_single = hard_class_acc(single_top1)
    hard_acc_tta = hard_class_acc(tta_top1)

    # ---------------- per-class P/R/F1 (from TTA confusion matrix — the
    # number you'd actually deploy with) ----------------
    rows = []
    for i, cls in enumerate(class_names):
        support = cm_tta[i, :].sum()
        tp = cm_tta[i, i]
        pred_total = cm_tta[:, i].sum()
        precision = tp / pred_total if pred_total > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append({'class': cls, 'support': int(support), 'precision': round(precision, 4),
                      'recall': round(recall, 4), 'f1': round(f1, 4)})
    pcm_df = pd.DataFrame(rows)
    pcm_df.to_csv(out_dir / 'per_class_metrics.csv', index=False)

    present = pcm_df[pcm_df['support'] > 0]
    macro_f1 = present['f1'].mean() if len(present) else 0.0
    n_zero_support = (pcm_df['support'] == 0).sum()

    # ---------------- confusion matrix + confused pairs (TTA) ----------------
    pd.DataFrame(cm_tta, index=class_names, columns=class_names).to_csv(out_dir / 'confusion_matrix.csv')

    pair_rows = []
    for i, true_cls in enumerate(class_names):
        for j, pred_cls in enumerate(class_names):
            if i != j and cm_tta[i, j] > 0:
                pair_rows.append({'true_class': true_cls, 'predicted_class': pred_cls,
                                   'count': int(cm_tta[i, j])})
    pairs_df = pd.DataFrame(pair_rows, columns=['true_class', 'predicted_class', 'count'])
    pairs_df = pairs_df.sort_values('count', ascending=False)
    pairs_df.to_csv(out_dir / 'top_confused_pairs.csv', index=False)

    grouped_rows = []
    if len(pairs_df):
        totals = pairs_df.groupby('true_class')['count'].sum().sort_values(ascending=False)
        g = pairs_df.groupby('true_class')
        for true_cls in totals.index:
            sub = g.get_group(true_cls).sort_values('count', ascending=False).head(5)
            first = True
            for _, r in sub.iterrows():
                grouped_rows.append({'true_class': true_cls if first else '',
                                      'misspredicted_class': f"{r['predicted_class']} - {int(r['count'])}"})
                first = False
    with open(out_dir / 'top_confusions_per_class.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['true_class', 'misspredicted_class'])
        w.writeheader(); w.writerows(grouped_rows)

    # ---------------- CUGAR-specific: per-cluster accuracy + confusion-pair
    # breakdown (Correct / same-cluster near-miss / different-cluster
    # far-miss), using TTA predictions ----------------
    cluster_rows = []
    for cid, clset in enumerate(CONFUSION_CLUSTERS):
        member_idx = [class_to_idx[c] for c in clset if c in class_to_idx]
        mask = np.isin(all_labels, member_idx)
        support = int(mask.sum())
        if support == 0:
            continue
        acc = 100 * (tta_top1[mask] == all_labels[mask]).mean()
        cluster_rows.append({
            'cluster_id': cid, 'members': ', '.join(sorted(clset)),
            'support': support, 'accuracy': round(acc, 2),
        })
    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df.to_csv(out_dir / 'cluster_accuracy.csv', index=False)

    def confusion_pair_breakdown(preds, mask=None):
        idx = np.arange(n) if mask is None else np.where(mask)[0]
        correct = same_cluster_wrong = diff_cluster_wrong = 0
        for i in idx:
            t, p = all_labels[i], preds[i]
            if t == p:
                correct += 1
            elif cluster_of.get(t) is not None and cluster_of.get(t) == cluster_of.get(p):
                same_cluster_wrong += 1
            else:
                diff_cluster_wrong += 1
        total = len(idx)
        return {
            'total': total,
            'correct_pct': round(100 * correct / total, 2) if total else 0.0,
            'same_cluster_wrong_pct': round(100 * same_cluster_wrong / total, 2) if total else 0.0,
            'different_cluster_wrong_pct': round(100 * diff_cluster_wrong / total, 2) if total else 0.0,
        }

    pair_breakdown_all = confusion_pair_breakdown(tta_top1)
    hard_mask = np.isin(all_labels, list(hard_indices))
    pair_breakdown_hard = confusion_pair_breakdown(tta_top1, mask=hard_mask)

    # ---------------- summary ----------------
    summary = {
        'n_eval_samples': int(n),
        'top1_accuracy_single': round(float(top1_single), 2),
        'top1_accuracy_tta': round(float(top1_tta), 2),
        'top5_accuracy_single': round(float(top5_single), 2),
        'top5_accuracy_tta': round(float(top5_tta), 2),
        'tta_gain_top1_pts': round(float(top1_tta - top1_single), 2),
        'macro_f1_tta': round(float(macro_f1), 4),
        'classes_with_zero_support_in_eval_set': int(n_zero_support),
        'hard_class_accuracy_single': round(float(hard_acc_single), 2),
        'hard_class_accuracy_tta': round(float(hard_acc_tta), 2),
        'confusion_pair_breakdown_all_classes_tta': pair_breakdown_all,
        'confusion_pair_breakdown_hard_classes_only_tta': pair_breakdown_hard,
    }
    with open(out_dir / 'overall_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ---------------- console printout ----------------
    print("\n" + "=" * 65)
    print(f"EVALUATION RESULTS  ({n} samples)")
    print("=" * 65)
    print(f"  Top-1 Accuracy       single: {top1_single:6.2f}%   TTA: {top1_tta:6.2f}%   "
          f"(TTA gain: {top1_tta - top1_single:+.2f} pts)")
    print(f"  Top-5 Accuracy       single: {top5_single:6.2f}%   TTA: {top5_tta:6.2f}%")
    print(f"  Macro F1 (TTA)       {macro_f1:.4f}  "
          f"({n_zero_support} of {num_classes} classes had 0 support in this eval set, excluded)")
    print(f"  Hard-class Accuracy  single: {hard_acc_single:6.2f}%   TTA: {hard_acc_tta:6.2f}%   "
          f"({int(hard_mask.sum())} samples across {len(hard_indices)} hard classes)")
    print(f"  Confusion-pair breakdown (TTA, all classes):  "
          f"correct={pair_breakdown_all['correct_pct']}%  "
          f"same-cluster miss={pair_breakdown_all['same_cluster_wrong_pct']}%  "
          f"different-cluster miss={pair_breakdown_all['different_cluster_wrong_pct']}%")
    print(f"  Confusion-pair breakdown (TTA, hard classes only): "
          f"correct={pair_breakdown_hard['correct_pct']}%  "
          f"same-cluster miss={pair_breakdown_hard['same_cluster_wrong_pct']}%  "
          f"different-cluster miss={pair_breakdown_hard['different_cluster_wrong_pct']}%")
    print(f"\n  Per-cluster accuracy (TTA):")
    for r in cluster_rows:
        print(f"    cluster {r['cluster_id']:2d} (n={r['support']:4d}) acc={r['accuracy']:6.2f}%  "
              f"[{r['members']}]")
    print(f"\n  Worst 10 classes by F1 (TTA):")
    print(pcm_df.sort_values('f1').head(10).to_string(index=False))
    print(f"\n  Files written to {out_dir}:")
    for f in ['per_class_metrics.csv', 'confusion_matrix.csv', 'top_confused_pairs.csv',
              'top_confusions_per_class.csv', 'cluster_accuracy.csv', 'overall_summary.json']:
        print(f"    {f}")
    print("=" * 65 + "\n")

    # ---------------- optional confusion-matrix heatmap PNG ----------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(20, 20))
        im = ax.imshow(np.log1p(cm_tta), cmap='viridis')
        ax.set_xticks(range(num_classes)); ax.set_xticklabels(class_names, rotation=90, fontsize=5)
        ax.set_yticks(range(num_classes)); ax.set_yticklabels(class_names, fontsize=5)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title('Confusion Matrix (log-scaled counts, TTA)')
        fig.colorbar(im, fraction=0.03)
        fig.tight_layout()
        fig.savefig(out_dir / 'confusion_matrix_heatmap.png', dpi=150)
        plt.close(fig)
        print(f"  Confusion matrix heatmap saved to {out_dir / 'confusion_matrix_heatmap.png'}")
    except ImportError:
        print("  (matplotlib not installed — skipped confusion_matrix_heatmap.png, "
              "CSV is still written)")

    return summary


# ============================================================================
# MAIN
# ============================================================================

CHECKPOINT_PATH = r'C:\Users\Amma.DESKTOP-4K4SV7F\Documents\Hemanth\fine-grain\cugar_jets_best.pth'
TEST_DIR = r'C:\Users\Amma.DESKTOP-4K4SV7F\Documents\Hemanth\fine-grain\test-fgvc\train'
OUT_DIR = r'C:\Users\Amma.DESKTOP-4K4SV7F\Documents\Hemanth\fine-grain\reports\full_eval'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    classes = ckpt['classes']
    print(f"  Checkpoint has {len(classes)} classes, "
          f"trained val_acc_single={ckpt.get('val_acc_single', '?')}, "
          f"val_acc_tta={ckpt.get('val_acc_tta', '?')}")

    model = CUGARViT(num_classes=len(classes)).to(device)
    try:
        model.load_state_dict(ckpt['model_state_dict'])
    except RuntimeError as e:
        print(f"  *** strict load failed ({e}); retrying with strict=False "
              f"(expected if this checkpoint is from a different script version). ***")
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  missing keys: {missing}\n  unexpected keys: {unexpected}")
    model.eval()

    print(f"\nResolving class-id mapping for {TEST_DIR} ...")
    id_to_class, mapping_source = resolve_class_id_mapping(TEST_DIR, classes)
    print(f"  Class-id mapping sample (VERIFY THIS LOOKS RIGHT):")
    for i in [0, 1, 2, 48, len(id_to_class) - 1]:
        if 0 <= i < len(id_to_class):
            print(f"    id {i:3d} -> {id_to_class[i]}")

    dataset = YoloFlatTestDataset(TEST_DIR, id_to_class=id_to_class)
    if len(dataset) == 0:
        print("\n*** 0 usable samples — nothing to evaluate. Check the skip counts "
              "above and the class-id mapping printed before them. ***")
        return

    evaluate_full(model, dataset, class_names=id_to_class, device=device, out_dir=OUT_DIR)


if __name__ == '__main__':
    main()