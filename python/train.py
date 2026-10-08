"""품종별 폴더의 사진으로 분류기를 학습한다.

    python train.py --data dataset/            # dataset/<품종>/<사진>.jpg
"""
import argparse
import os
import numpy as np
import joblib
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from common import Clip, DEFAULT_MODEL, IMG_EXT, hybrid_probs, softmax


def scan(root, min_per_class):
    classes, paths, labels = [], [], []
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            continue
        files = [os.path.join(full, f) for f in sorted(os.listdir(full))
                 if os.path.splitext(f)[1].lower() in IMG_EXT]
        if len(files) < min_per_class:
            print(f"건너뜀: {d} ({len(files)}장, 최소 {min_per_class}장 필요)")
            continue
        classes.append(d.replace("_", " ").replace("-", " ").strip())
        paths += files
        labels += [len(classes) - 1] * len(files)
    return classes, paths, np.array(labels)


def embed_all(clip, paths, batch):
    from tqdm import tqdm
    feats, ok = [], []
    for i in tqdm(range(0, len(paths), batch), desc="사진 분석"):
        imgs, idx = [], []
        for j, p in enumerate(paths[i:i + batch]):
            try:
                imgs.append(Image.open(p).convert("RGB")); idx.append(i + j)
            except Exception:
                print("읽기 실패:", p)
        if imgs:
            feats.append(clip.images(imgs)); ok += idx
    return np.concatenate(feats), np.array(ok)


def fit_clf(x, y, C):
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(x, y)


def topk_acc(probs, y, k):
    top = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y[i] in top[i] for i in range(len(y))]))


def tune(xtr, ytr, xva, yva, text, n, C):
    """검증 데이터로 alpha(문장 비중)를 고른다."""
    clf = fit_clf(xtr, ytr, C)
    zs = softmax(100.0 * xva @ text.T)
    print(f"제로샷(문장만)   top1 {np.mean(zs.argmax(1) == yva):.3f}  top3 {topk_acc(zs, yva, 3):.3f}")
    best_a, best = 0.0, -1
    for a in np.linspace(0, 1, 11):
        p = hybrid_probs(xva, text, clf, a, n)
        acc = np.mean(p.argmax(1) == yva)
        print(f"alpha {a:.1f}        top1 {acc:.3f}  top3 {topk_acc(p, yva, 3):.3f}")
        if acc > best + 1e-9:
            best, best_a = acc, float(a)
    print(f"-> 선택한 alpha = {best_a:.1f} (검증 top1 {best:.3f})")
    return best_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="품종별 하위 폴더가 있는 폴더")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="cat_model.joblib")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--min-per-class", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--cache", default="features.npz")
    a = ap.parse_args()

    classes, paths, labels = scan(a.data, a.min_per_class)
    print(f"품종 {len(classes)}개, 사진 {len(paths)}장")
    if len(classes) < 2:
        raise SystemExit("품종 폴더가 2개 이상 필요합니다.")

    clip = Clip(a.model)
    if os.path.exists(a.cache):
        c = np.load(a.cache, allow_pickle=True)
        if list(c["paths"]) == paths and str(c["model"]) == a.model:
            feats, keep = c["feats"], np.arange(len(paths))
            print("저장된 특징 사용:", a.cache)
        else:
            c = None
    else:
        c = None
    if c is None:
        feats, keep = embed_all(clip, paths, a.batch)
        np.savez(a.cache, feats=feats, paths=np.array(paths), model=a.model)
    y = labels[keep]
    text = clip.class_text_embeds(classes)

    xtr, xva, ytr, yva = train_test_split(feats, y, test_size=a.val, stratify=y, random_state=0)
    alpha = tune(xtr, ytr, xva, yva, text, len(classes), a.C)

    clf = fit_clf(feats, y, a.C)  # 최종 모델은 전체 사진으로 다시 학습
    joblib.dump(dict(classes=classes, clf=clf, text=text, alpha=alpha, model=a.model), a.out)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
