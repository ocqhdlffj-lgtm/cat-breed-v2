"""학습한 모델로 사진 한 장 판별:  python predict.py cat.jpg"""
import sys
import joblib
from PIL import Image
from common import Clip, hybrid_probs
from breeds import KO


def load(path="cat_model.joblib"):
    b = joblib.load(path)
    b["clip"] = Clip(b["model"])
    return b


def predict(b, img, k=5):
    f = b["clip"].images([img.convert("RGB")])
    p = hybrid_probs(f, b["text"], b["clf"], b["alpha"], len(b["classes"]))[0]
    order = p.argsort()[::-1][:k]
    return [(b["classes"][i], float(p[i])) for i in order]


if __name__ == "__main__":
    model = load(sys.argv[2] if len(sys.argv) > 2 else "cat_model.joblib")
    for name, p in predict(model, Image.open(sys.argv[1])):
        print(f"{KO.get(name, name):<20} {p*100:5.1f}%")
