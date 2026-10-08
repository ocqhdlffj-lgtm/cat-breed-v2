"""인터넷이 되는 PC에서 한 번만 실행하면 cat-breed-embedded.html (파일 하나) 이 만들어집니다.

    python build_embedded.py

만들어진 HTML은 인터넷 없이도 열립니다. (모델·라이브러리가 모두 파일 안에 들어 있어 약 200MB)
표준 라이브러리만 사용합니다.
"""
import base64, gzip, io, json, os, sys, tarfile, urllib.request

REPO = "Xenova/clip-vit-base-patch32"
NPM_TGZ = "https://registry.npmjs.org/@huggingface/transformers/-/transformers-3.5.1.tgz"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

REQUIRED = ["config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
            "onnx/text_model_quantized.onnx", "onnx/vision_model_quantized.onnx"]
OPTIONAL = ["special_tokens_map.json", "vocab.json", "merges.txt"]


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("내려받는 중:", url)
    req = urllib.request.Request(url, headers={"User-Agent": "catbreed-build"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(path + ".part", path)


def fetch_library():
    tgz = os.path.join(CACHE, "transformers-3.5.1.tgz")
    download(NPM_TGZ, tgz)
    want = {"package/dist/transformers.min.js": "lib.js",
            "package/dist/ort-wasm-simd-threaded.jsep.mjs": "ort.mjs",
            "package/dist/ort-wasm-simd-threaded.jsep.wasm": "ort.wasm"}
    out = {}
    with tarfile.open(tgz) as t:
        for m in t.getmembers():
            if m.name in want:
                out[want[m.name]] = t.extractfile(m).read()
    missing = set(want.values()) - set(out)
    if missing:
        sys.exit(f"라이브러리 파일을 찾지 못했습니다: {missing}")
    return out


def fetch_model():
    api = os.path.join(CACHE, "model_info.json")
    download(f"https://huggingface.co/api/models/{REPO}", api)
    siblings = {s["rfilename"] for s in json.load(open(api))["siblings"]}
    for f in REQUIRED:
        if f not in siblings:
            onnx = sorted(x for x in siblings if x.startswith("onnx/"))
            sys.exit(f"모델 저장소에 {f} 가 없습니다. 있는 onnx 파일: {onnx}")
    out = {}
    for f in REQUIRED + [o for o in OPTIONAL if o in siblings]:
        p = os.path.join(CACHE, "hf", f)
        download(f"https://huggingface.co/{REPO}/resolve/main/{f}", p)
        out["hf/" + f] = open(p, "rb").read()
    return out


def assemble(files, template=os.path.join(HERE, "template.html"), out_path=None):
    tags = []
    for name, data in files.items():
        b64 = base64.b64encode(gzip.compress(data, 6)).decode("ascii")
        tags.append(f'<script type="application/octet-stream" class="emb" data-name="{name}">{b64}</script>')
        print(f"포함: {name}  {len(data)/1e6:.1f}MB -> {len(b64)/1e6:.1f}MB")
    html = open(template, encoding="utf-8").read()
    assert "<!--EMBEDDED_DATA-->" in html
    html = html.replace("<!--EMBEDDED_DATA-->", "\n".join(tags))
    out_path = out_path or os.path.join(HERE, "cat-breed-embedded.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"완료: {out_path}  ({os.path.getsize(out_path)/1e6:.0f}MB)")
    return out_path


if __name__ == "__main__":
    files = fetch_library()
    files.update(fetch_model())
    assemble(files)
