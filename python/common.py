"""CLIP 임베딩과 하이브리드 점수 계산 (train.py, predict.py, app.py가 함께 사용)."""
import numpy as np

TEMPLATES = [
    "a photo of a {} cat.", "a close-up photo of a {} cat.", "a portrait of a {} cat.",
    "a high quality photo of a {} cat.", "a {} cat sitting indoors.", "a {} cat outdoors.", "a cute {} cat.",
]
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MODEL = "openai/clip-vit-base-patch32"  # 더 정확하게: openai/clip-vit-large-patch14


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class Clip:
    def __init__(self, name=DEFAULT_MODEL, device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)

    def _out(self, f):
        return f if hasattr(f, "shape") else f.pooler_output

    def images(self, pil_images):
        torch = self.torch
        with torch.no_grad():
            inp = self.proc(images=pil_images, return_tensors="pt").to(self.device)
            f = self._out(self.model.get_image_features(**inp))
            return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()

    def texts(self, texts):
        torch = self.torch
        with torch.no_grad():
            inp = self.proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            f = self._out(self.model.get_text_features(**inp))
            return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()

    def class_text_embeds(self, names):
        """품종마다 문장 7개의 평균 임베딩을 만든다."""
        out = []
        for n in names:
            base = n[:-4] if n.lower().endswith(" cat") else n  # 템플릿이 이미 "cat"을 붙임
            e = self.texts([t.format(base) for t in TEMPLATES]).mean(axis=0)
            out.append(e / np.linalg.norm(e))
        return np.stack(out)


def hybrid_probs(feats, text_embeds, clf, alpha, n_classes, temp=100.0):
    """alpha * (문장 기반 제로샷) + (1 - alpha) * (사진으로 학습한 분류기)"""
    zs = softmax(temp * feats @ text_embeds.T)
    pc = np.zeros((len(feats), n_classes))
    pc[:, clf.classes_] = clf.predict_proba(feats)
    return alpha * zs + (1 - alpha) * pc
