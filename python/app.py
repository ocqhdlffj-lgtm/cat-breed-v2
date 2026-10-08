"""웹 화면 실행:  python app.py  → 브라우저에서 http://127.0.0.1:7860"""
import gradio as gr
from predict import load, predict
from breeds import KO

model = load()


def run(img):
    if img is None:
        return {}
    return {KO.get(n, n): p for n, p in predict(model, img)}


gr.Interface(run, gr.Image(type="pil", label="고양이 사진"), gr.Label(num_top_classes=5, label="판별 결과"),
             title="고양이 품종 판별기").launch()
