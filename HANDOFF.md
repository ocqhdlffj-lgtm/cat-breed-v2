# 고양이 품종 판별기 — 인계 문서 (Claude Code용)

작성일: 2026-10-07. 이 문서 하나에 요구사항, 결정 사항, 구조, 전체 코드, 남은 과제를 담았습니다.
아래 "프로젝트 파일" 절의 코드 블록을 각 경로에 저장하면 프로젝트가 그대로 복원됩니다.

## 1. 목표

사용자가 고양이 사진을 넣으면 **어떤 품종인지 판별하는 앱**. 요구는 "세계 모든 고양이".
- 한국어 사용자. 화면·메시지는 한국어.
- 사진은 서버로 보내지 않고 브라우저(또는 로컬 PC)에서만 처리.
- 사용자는 이후 **코드 작업(Claude Code, 로컬 PC)** 으로 넘겨 계속 개발할 계획.

## 2. 지금까지의 진행

| 단계 | 내용 | 상태 |
|---|---|---|
| v1 | 브라우저 단독 HTML. transformers.js + CLIP(`Xenova/clip-vit-base-patch32`) 제로샷 분류 | 완료(문법 검사만) |
| v2-웹 | 문장 7개 평균(프롬프트 앙상블) + 품종별 기준 사진 폴더 비교(IndexedDB 저장) | 완료(문법 검사만) |
| v2-Python | 기준 사진으로 로지스틱 회귀 학습 + 제로샷 혼합(alpha 자동 선택), CLI·Gradio 앱 | 완료(가짜 데이터로 로직 검증) |
| 단일 HTML | 모델·라이브러리를 HTML 안에 내장하는 빌드 스크립트 | 완료(가짜 모델로 오프라인 기동 검증) |

**중요: 실제 CLIP 모델과 실제 고양이 사진으로는 한 번도 실행해 보지 못했습니다.** 작업 환경에서 huggingface.co 접속이 막혀 있었습니다. 첫 작업은 로컬에서 실제로 돌려 보는 것입니다.

## 3. 핵심 설계 결정과 이유

1. **CLIP 제로샷을 기본으로 사용.** 품종 전용 학습 데이터 없이도 71개 항목(순수 품종 63 + 믹스 2 + 털 무늬 6)을 비교할 수 있음. 품종 목록은 `BREEDS` 배열(웹)과 `python/breeds.py`(동일 내용)에 있음.
2. **고양이가 아닌 입력 걸러내기.** `dog, person, landscape, object, bird` 5개 항목을 함께 비교하고, 1순위가 이 중 하나면 "고양이가 아닌 것 같아요" 표시.
3. **프롬프트 앙상블.** 품종마다 문장 7개(`TEMPLATES`)의 임베딩을 평균 후 정규화.
4. **기준 사진 방식(웹).** 품종별 최상위 3장 평균 유사도 → softmax(온도 100) → 제로샷 확률과 혼합. 기준 사진이 있는 품종들의 제로샷 확률 총합(`mass`)을 보존하며 `REF_WEIGHT=0.7`로 섞음. 기준 사진이 없는 품종 점수는 그대로 둠.
5. **학습 방식(Python).** CLIP 이미지 임베딩 → LogisticRegression(`class_weight=balanced`, C=10). 검증 데이터로 `alpha`(제로샷 비중, 0~1)를 골라 `alpha*제로샷 + (1-alpha)*분류기` 사용. 최종 모델은 전체 사진으로 재학습.
6. **단일 HTML 내장.** 모델 가중치(양자화 q8 onnx) 포함 약 200MB. 라이브러리는 번들된 `transformers.min.js`를 사용해야 함(`transformers.web.min.js`는 `onnxruntime-common`을 외부 import라 단독 파일에서 실패 — 실제로 겪은 문제). `window.fetch`를 가로채 `/resolve/<rev>/<파일>` 요청을 내장 데이터로 응답. wasm은 `env.backends.onnx.wasm.wasmPaths = {mjs, wasm}`에 blob URL, `numThreads=1`, `proxy=false`.
7. **공유 가능한 링크(Claude 아티팩트)로는 못 만듦.** 아티팩트 CSP가 외부 모델 다운로드를 막음. 그래서 로컬 파일 형태로 전달.

## 4. 알려진 한계

- 믹스묘·길고양이는 가장 비슷한 품종으로 표시됨(정답 개념 없음).
- 닮은 품종 혼동: 브리티시 숏헤어↔러시안 블루, 메인쿤↔노르웨이 숲, 샴↔발리니즈 등.
- 작은 CLIP(ViT-B/32) 기준. 더 큰 모델(`openai/clip-vit-large-patch14`)이 더 정확하지만 느리고 큼.
- 외관 추정일 뿐 혈통 확인이 아님. 화면에 면책 문구 있음.
- 단일 HTML은 열 때 압축 해제에 10~30초, 메모리를 많이 씀(데스크톱 크롬/엣지 권장).

## 5. 다음에 할 일 (우선순위 순)

1. **실제 실행 검증**: `web/index.html`을 크롬으로 열어 고양이 사진 몇 장 테스트. `python/`은 소량 데이터셋으로 `train.py` 실행.
2. **데이터셋 확보**: 품종별 폴더(`dataset/<영문 품종명>/*.jpg`), 품종당 30장 이상 권장. Kaggle 등 고양이 품종 데이터셋 사용(라이선스 확인). Oxford-IIIT Pet은 고양이 12품종뿐.
3. **정확도 측정·개선**: `train.py`의 검증 출력(제로샷 vs alpha별)으로 효과 확인. 필요하면 큰 CLIP 모델, 사진 증강, 파인튜닝(선형 프로브 이상) 검토.
4. **`build_embedded.py` 실제 실행**: 인터넷 되는 PC에서 실행해 단일 HTML 생성 후 오프라인 동작 확인. 모델 저장소의 파일명이 다르면 스크립트가 있는 onnx 목록을 출력하고 종료함.
5. **웹과 Python의 품종 목록 동기화**: 지금은 복사본 2개. 하나의 JSON으로 합치는 것 권장.
6. (선택) 학습된 분류기를 웹에서 쓰도록 내보내기(선형 가중치를 JSON으로 저장 후 웹에서 로드).

## 6. 실행 방법

### 웹 (설치 없음)
`web/index.html`을 크롬/엣지로 열기. 첫 실행 시 모델 약 200MB를 내려받음(인터넷 필요).
"기준 사진으로 정확도 높이기"에서 품종별 폴더를 선택하면 반영됨(폴더 이름 = 품종명, 영문/한글 모두 가능, 목록에 없으면 새 품종).

### Python
```
cd python
pip install -r requirements.txt
python train.py --data dataset/            # 검증 정확도 출력, cat_model.joblib 저장
python predict.py cat.jpg                  # 한 장 판별
python app.py                              # http://127.0.0.1:7860
```
옵션: `--model openai/clip-vit-large-patch14`, `--val 0.2`, `--C 10`, `--min-per-class 5`, `--cache features.npz`.
주의: `requirements.txt`에 `transformers>=4.40,<5`로 고정(`get_image_features` 반환형 때문). `predict.py`/`app.py`는 저장된 모델 파일에 기록된 CLIP 모델명을 자동 사용.

### 단일 HTML 생성
```
cd embed
python build_embedded.py      # huggingface.co, registry.npmjs.org 접속 필요. 결과: cat-breed-embedded.html
```

## 7. 프로젝트 파일

```
cat-breed-v2/
  web/index.html
  python/{common.py, train.py, predict.py, app.py, breeds.py, requirements.txt}
  embed/{build_embedded.py, template.html}
```

`embed/template.html`은 `web/index.html`에서 아래 변경만 적용한 파일입니다(7-4 참고).

## 7-1. 웹 앱

### `web/index.html`

````html
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>고양이 품종 판별기</title>
<style>
  :root{--bg:#f5f3ee;--card:#fff;--fg:#1d1b18;--mute:#6f6a62;--line:#e1ddd4;--accent:#c4501b;--bar:#e9e4d9}
  @media(prefers-color-scheme:dark){:root{--bg:#161412;--card:#201d1a;--fg:#f1ede6;--mute:#a39d92;--line:#37332e;--accent:#ee8a55;--bar:#2e2a26}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 "Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic",system-ui,sans-serif}
  main{max-width:640px;margin:0 auto;padding:32px 16px 64px}
  h1{font-size:1.6rem;margin:0 0 4px}
  .sub{color:var(--mute);margin:0 0 24px}
  #drop{display:block;border:2px dashed var(--line);border-radius:16px;background:var(--card);min-height:260px;cursor:pointer;overflow:hidden;position:relative}
  #drop.over,#drop:focus-visible{border-color:var(--accent);outline:none}
  #drop .hint{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;color:var(--mute);text-align:center;padding:16px}
  #drop .hint b{color:var(--fg)}
  #preview{display:none;width:100%;max-height:420px;object-fit:contain;background:#000}
  #status{margin:16px 0 8px;color:var(--mute);font-size:.9rem;min-height:1.4em;overflow-wrap:anywhere}
  progress{width:100%;height:6px;accent-color:var(--accent);display:none;margin-bottom:8px}
  #result{display:none;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;margin-top:8px}
  #top{font-size:1.5rem;font-weight:700;margin:0}
  #top small{display:block;font-size:.85rem;font-weight:400;color:var(--mute)}
  ol{list-style:none;margin:16px 0 0;padding:0;display:grid;gap:10px}
  li{display:grid;grid-template-columns:1fr auto;gap:4px 12px;font-size:.95rem}
  li .n{min-width:0;overflow-wrap:anywhere}
  li .p{font-variant-numeric:tabular-nums;color:var(--mute)}
  li .b{grid-column:1/-1;height:6px;border-radius:3px;background:var(--bar);overflow:hidden}
  li .b i{display:block;height:100%;background:var(--accent)}
  .note{color:var(--mute);font-size:.8rem;margin-top:20px}
  details{margin-top:24px;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:14px 20px}
  summary{cursor:pointer;font-weight:600}
  details p,details pre{font-size:.88rem;color:var(--mute);margin:10px 0}
  details pre{background:var(--bar);padding:10px 12px;border-radius:8px;overflow-x:auto;color:var(--fg)}
  .row{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
  button,.btn{font:inherit;font-size:.9rem;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer}
  button:hover,.btn:hover{border-color:var(--accent)}
  button:disabled{opacity:.5;cursor:default}
  #refinfo{font-size:.88rem;margin-top:10px}
</style>
</head>
<body>
<main>
  <h1>고양이 품종 판별기</h1>
  <p class="sub">고양이 사진을 올리면 어떤 품종에 가장 가까운지 알려줍니다. 사진은 서버로 전송되지 않고 내 브라우저에서만 처리됩니다.</p>

  <label id="drop" for="file" tabindex="0">
    <img id="preview" alt="업로드한 고양이 사진">
    <div class="hint" id="hint"><b>사진을 끌어다 놓거나 눌러서 선택</b><span>JPG, PNG, WEBP</span></div>
  </label>
  <input id="file" type="file" accept="image/*" hidden>

  <div id="status">모델을 준비하는 중입니다. 처음 한 번만 약 200MB를 내려받습니다.</div>
  <progress id="prog" max="100" value="0"></progress>

  <section id="result" aria-live="polite">
    <p id="top"></p>
    <ol id="list"></ol>
    <p class="note">AI가 외모로 추정한 결과이며 정확한 품종 확인은 수의사나 혈통서가 필요합니다. 믹스묘와 길고양이는 가장 비슷한 품종이 표시됩니다.</p>
  </section>

  <details id="refs">
    <summary>기준 사진으로 정확도 높이기 (선택)</summary>
    <p>품종별 폴더에 사진을 넣어 두고 폴더를 선택하면, 내 사진과 가장 비슷한 기준 사진을 찾아 결과에 반영합니다. 품종당 10~30장이면 충분합니다. 폴더 이름이 품종 이름이 됩니다. 영문(Persian)이나 한글(페르시안) 모두 됩니다. 목록에 없는 이름은 새 품종으로 추가됩니다.</p>
<pre>reference/
  Persian/   a.jpg b.jpg ...
  Siamese/   ...
  코리안 숏헤어/ ...</pre>
    <div class="row">
      <label class="btn" for="refdir" id="refbtn" tabindex="0">폴더 선택</label>
      <input id="refdir" type="file" webkitdirectory multiple hidden>
      <button id="refclear" type="button">기준 사진 지우기</button>
    </div>
    <div id="refinfo">불러온 기준 사진: 없음</div>
  </details>
</main>

<script type="module">
import { AutoTokenizer, AutoProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, RawImage }
  from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1";

const MODEL = "Xenova/clip-vit-base-patch32";
const MAX_PER_BREED = 40;   // 품종당 최대 기준 사진 수
const REF_WEIGHT = 0.7;     // 기준 사진 결과를 반영하는 비율

// [영문명, 한글명]
const BREEDS = [
 ["Abyssinian","아비시니안"],["American Bobtail","아메리칸 밥테일"],["American Curl","아메리칸 컬"],["American Shorthair","아메리칸 숏헤어"],
 ["American Wirehair","아메리칸 와이어헤어"],["Balinese","발리니즈"],["Bengal","벵갈"],["Birman","버만"],["Bombay","봄베이"],
 ["British Longhair","브리티시 롱헤어"],["British Shorthair","브리티시 숏헤어"],["Burmese","버미즈"],["Burmilla","버밀라"],
 ["Chartreux","샤르트뢰"],["Chausie","차우시"],["Cornish Rex","콘월 렉스"],["Cymric","킴릭"],["Devon Rex","데본 렉스"],
 ["Donskoy","돈스코이"],["Egyptian Mau","이집션 마우"],["European Burmese","유러피안 버미즈"],["Exotic Shorthair","엑조틱 숏헤어"],
 ["Havana Brown","하바나 브라운"],["Highlander","하이랜더"],["Himalayan","히말라얀"],["Japanese Bobtail","재패니즈 밥테일"],
 ["Javanese","자바니즈"],["Khao Manee","카오마니"],["Korat","코랫"],["Kurilian Bobtail","쿠릴리안 밥테일"],["LaPerm","라펌"],
 ["Lykoi","라이코이"],["Maine Coon","메인쿤"],["Manx","맹크스"],["Minskin","민스킨"],["Munchkin","먼치킨"],
 ["Nebelung","네벨룽"],["Norwegian Forest Cat","노르웨이 숲 고양이"],["Ocicat","오시캣"],["Oriental Shorthair","오리엔탈 숏헤어"],
 ["Persian","페르시안"],["Peterbald","피터볼드"],["Pixie-bob","픽시밥"],["Ragamuffin","라가머핀"],["Ragdoll","랙돌"],
 ["Russian Blue","러시안 블루"],["Savannah","사바나"],["Scottish Fold","스코티시 폴드"],["Selkirk Rex","셀커크 렉스"],
 ["Serengeti","세렝게티"],["Siamese","샴"],["Siberian","시베리안"],["Singapura","싱가푸라"],["Snowshoe","스노우슈"],
 ["Somali","소말리"],["Sphynx","스핑크스"],["Tonkinese","통키니즈"],["Toyger","토이거"],["Turkish Angora","터키시 앙고라"],
 ["Turkish Van","터키시 반"],["Ukrainian Levkoy","우크라이니안 레프코이"],["York Chocolate","요크 초콜릿"],
 ["Korean Shorthair","코리안 숏헤어"],["domestic shorthair","도메스틱 숏헤어(믹스)"],["domestic longhair","도메스틱 롱헤어(믹스)"],
 ["orange tabby","치즈 태비"],["tuxedo","턱시도"],["calico","칼리코(삼색이)"],["tortoiseshell","트라이컬러(고등어 삼색)"],
 ["silver tabby","실버 태비"],["mackerel tabby","고등어 태비"]
];
const NOT_CAT = [["dog","개"],["person","사람"],["landscape","풍경"],["object","물체"],["bird","새"]];

// 문장 여러 개의 평균으로 품종 설명을 만든다.
const TEMPLATES = [
  "a photo of a {}.", "a close-up photo of a {}.", "a portrait of a {}.",
  "a high quality photo of a {}.", "a {} sitting indoors.", "a {} outdoors.", "a cute {}."
];

const $ = id => document.getElementById(id);
const status = t => { $("status").textContent = t; };
const prog = $("prog");
const setProg = v => { if (v == null) prog.style.display = "none"; else { prog.style.display = "block"; prog.value = v; } };

// ---------- 벡터 도구 ----------
const norm = v => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i]*v[i]; s = Math.sqrt(s) || 1; for (let i = 0; i < v.length; i++) v[i] /= s; return v; };
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]*b[i]; return s; };
const rows = t => { const [n, d] = t.dims; const out = []; for (let i = 0; i < n; i++) out.push(norm(Float32Array.from(t.data.subarray(i*d, (i+1)*d)))); return out; };
const softmax = a => { const m = Math.max(...a); const e = a.map(x => Math.exp(x - m)); const s = e.reduce((p, c) => p + c, 0); return e.map(x => x / s); };
const key = s => s.toLowerCase().replace(/[^a-z0-9가-힣]/g, "");

// ---------- 모델 ----------
let tokenizer, textModel, processor, visionModel;
async function loadModels() {
  const cb = p => { if (p.status === "progress") setProg(p.progress); };
  status("모델을 내려받는 중입니다 (텍스트)…");
  tokenizer = await AutoTokenizer.from_pretrained(MODEL);
  textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL, { progress_callback: cb });
  status("모델을 내려받는 중입니다 (이미지)…");
  processor = await AutoProcessor.from_pretrained(MODEL);
  visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL, { progress_callback: cb });
  setProg(null);
}
async function embedTexts(texts) {
  const out = [];
  for (let i = 0; i < texts.length; i += 48) {
    const inp = tokenizer(texts.slice(i, i + 48), { padding: true, truncation: true });
    const { text_embeds } = await textModel(inp);
    out.push(...rows(text_embeds));
  }
  return out;
}
async function embedBlob(blob) {
  const img = await RawImage.fromBlob(blob);
  const inp = await processor(img);
  const { image_embeds } = await visionModel(inp);
  return rows(image_embeds)[0];
}

// ---------- 라벨 ----------
// label: {key, en, ko, isCat, text(Float32Array), refs: Float32Array[]}
let labels = [];
const mkPrompts = (en, isCat) => TEMPLATES.map(t => t.replace("{}", isCat ? en + " cat" : en));
async function addTextEmbeds(list) {
  const flat = list.flatMap(l => mkPrompts(l.en, l.isCat));
  const emb = await embedTexts(flat);
  let k = 0;
  for (const l of list) {
    const acc = new Float32Array(emb[0].length);
    for (let i = 0; i < TEMPLATES.length; i++, k++) for (let j = 0; j < acc.length; j++) acc[j] += emb[k][j];
    l.text = norm(acc);
  }
}
async function buildLabels() {
  labels = [
    ...BREEDS.map(([en, ko]) => ({ key: en, en, ko, isCat: true, refs: [] })),
    ...NOT_CAT.map(([en, ko]) => ({ key: en, en, ko, isCat: false, refs: [] }))
  ];
  status("품종 설명을 계산하는 중입니다…");
  await addTextEmbeds(labels);
}
function findLabel(folder) {
  const k = key(folder);
  return labels.find(l => l.isCat && (key(l.en) === k || key(l.ko) === k || key(l.key) === k));
}

// ---------- 저장소(IndexedDB, 실패해도 동작) ----------
const idb = {
  open: () => new Promise((res, rej) => { const r = indexedDB.open("catbreed", 1); r.onupgradeneeded = () => r.result.createObjectStore("kv"); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); }),
  async get(k) { const db = await this.open(); return new Promise((res, rej) => { const q = db.transaction("kv").objectStore("kv").get(k); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); }); },
  async set(k, v) { const db = await this.open(); return new Promise((res, rej) => { const t = db.transaction("kv", "readwrite"); t.objectStore("kv").put(v, k); t.oncomplete = () => res(); t.onerror = () => rej(t.error); }); },
  async del(k) { const db = await this.open(); return new Promise((res, rej) => { const t = db.transaction("kv", "readwrite"); t.objectStore("kv").delete(k); t.oncomplete = () => res(); t.onerror = () => rej(t.error); }); }
};
async function saveRefs() {
  try {
    const data = labels.filter(l => l.refs.length).map(l => ({ key: l.key, en: l.en, ko: l.ko, custom: !BREEDS.some(b => b[0] === l.key), refs: l.refs }));
    await idb.set("refs", data);
  } catch (e) { /* 저장 실패는 무시 */ }
}
async function loadRefs() {
  try {
    const data = await idb.get("refs");
    if (!data) return;
    const customs = [];
    for (const d of data) {
      let l = labels.find(x => x.key === d.key);
      if (!l) { l = { key: d.key, en: d.en, ko: d.ko, isCat: true, refs: [] }; labels.push(l); customs.push(l); }
      l.refs = d.refs;
    }
    if (customs.length) await addTextEmbeds(customs);
  } catch (e) { /* 저장소를 못 읽어도 계속 */ }
}
function refSummary() {
  const withRefs = labels.filter(l => l.refs.length);
  const n = withRefs.reduce((s, l) => s + l.refs.length, 0);
  $("refinfo").textContent = n ? `불러온 기준 사진: ${withRefs.length}개 품종, ${n}장` : "불러온 기준 사진: 없음";
}

// ---------- 분류 ----------
function classify(emb) {
  const zs = softmax(labels.map(l => 100 * dot(emb, l.text)));
  const refIdx = labels.map((l, i) => l.refs.length ? i : -1).filter(i => i >= 0);
  if (!refIdx.length) return { probs: zs, used: 0 };
  // 기준 사진 점수: 가장 비슷한 최대 3장의 평균 유사도
  const sims = refIdx.map(i => {
    const s = labels[i].refs.map(r => dot(emb, r)).sort((a, b) => b - a).slice(0, 3);
    return s.reduce((p, c) => p + c, 0) / s.length;
  });
  const rp = softmax(sims.map(s => 100 * s));
  const mass = refIdx.reduce((s, i) => s + zs[i], 0);   // 기준 사진 품종들이 가진 확률 총합
  const out = zs.slice();
  refIdx.forEach((i, j) => { out[i] = (1 - REF_WEIGHT) * zs[i] + REF_WEIGHT * mass * rp[j]; });
  return { probs: out, used: refIdx.length };
}

async function run(file) {
  if (!file || !visionModel) return;
  const url = URL.createObjectURL(file);
  const img = $("preview"); img.src = url; img.style.display = "block"; $("hint").style.display = "none";
  status("분석 중…"); $("result").style.display = "none";
  try {
    const emb = await embedBlob(file);
    const { probs, used } = classify(emb);
    const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 5);
    const best = labels[order[0][1]];
    $("top").innerHTML = best.isCat
      ? best.ko + "<small>" + best.en + " · 확률 " + (order[0][0]*100).toFixed(1) + "%" + (used ? " · 기준 사진 " + used + "개 품종 반영" : "") + "</small>"
      : "고양이가 아닌 것 같아요<small>가장 가까운 항목: " + best.ko + "</small>";
    $("list").innerHTML = order.map(([p, i]) =>
      '<li><span class="n"></span><span class="p">' + (p*100).toFixed(1) + '%</span><span class="b"><i style="width:' + (p*100).toFixed(1) + '%"></i></span></li>').join("");
    [...$("list").children].forEach((li, k) => { li.querySelector(".n").textContent = labels[order[k][1]].ko; });
    $("result").style.display = "block";
    status("");
  } catch (e) { status("분석에 실패했습니다: " + e.message); }
}

// ---------- 기준 사진 불러오기 ----------
async function importRefs(files) {
  const imgs = [...files].filter(f => f.type.startsWith("image/"));
  const groups = new Map();
  for (const f of imgs) {
    const parts = (f.webkitRelativePath || "").split("/");
    if (parts.length < 2) continue;
    const folder = parts[parts.length - 2];
    if (!groups.has(folder)) groups.set(folder, []);
    groups.get(folder).push(f);
  }
  if (!groups.size) { status("이미지가 들어 있는 품종 폴더를 찾지 못했습니다."); return; }
  const total = [...groups.values()].reduce((s, g) => s + Math.min(g.length, MAX_PER_BREED), 0);
  let done = 0, failed = 0;
  const newOnes = [];
  for (const [folder, list] of groups) {
    let l = findLabel(folder);
    if (!l) { l = { key: folder, en: folder, ko: folder, isCat: true, refs: [] }; labels.push(l); newOnes.push(l); }
    l.refs = [];
    for (const f of list.slice(0, MAX_PER_BREED)) {
      try { l.refs.push(await embedBlob(f)); } catch (e) { failed++; }
      done++;
      status(`기준 사진 분석 중… ${done}/${total} (${folder})`);
      setProg(done / total * 100);
      if (done % 5 === 0) await new Promise(r => setTimeout(r));
    }
  }
  if (newOnes.length) await addTextEmbeds(newOnes);
  await saveRefs();
  setProg(null); refSummary();
  status(`기준 사진 ${done - failed}장을 반영했습니다.` + (failed ? ` (${failed}장은 읽지 못했습니다)` : ""));
}

// ---------- 시작 ----------
const dis = v => { $("file").disabled = v; $("refdir").disabled = v; $("refclear").disabled = v; };
dis(true);
try {
  await loadModels();
  await buildLabels();
  await loadRefs();
  refSummary();
  status("준비 완료. 고양이 사진을 올려 보세요.");
  dis(false);
} catch (e) {
  setProg(null);
  status("모델을 불러오지 못했습니다. 인터넷 연결을 확인하고 새로고침하세요. (" + e.message + ")");
}

$("file").addEventListener("change", e => run(e.target.files[0]));
$("refdir").addEventListener("change", async e => { dis(true); try { await importRefs(e.target.files); } finally { dis(false); e.target.value = ""; } });
$("refclear").addEventListener("click", async () => {
  labels = labels.filter(l => BREEDS.some(b => b[0] === l.key) || NOT_CAT.some(b => b[0] === l.key));
  labels.forEach(l => { l.refs = []; });
  try { await idb.del("refs"); } catch (e) {}
  refSummary(); status("기준 사진을 지웠습니다.");
});
const drop = $("drop");
drop.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
$("refbtn").addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("refdir").click(); } });
["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => run(e.dataTransfer.files[0]));
</script>
</body>
</html>
````

## 7-2. Python 학습 프로젝트

### `python/common.py`

````python
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
            e = self.texts([t.format(n) for t in TEMPLATES]).mean(axis=0)
            out.append(e / np.linalg.norm(e))
        return np.stack(out)


def hybrid_probs(feats, text_embeds, clf, alpha, n_classes, temp=100.0):
    """alpha * (문장 기반 제로샷) + (1 - alpha) * (사진으로 학습한 분류기)"""
    zs = softmax(temp * feats @ text_embeds.T)
    pc = np.zeros((len(feats), n_classes))
    pc[:, clf.classes_] = clf.predict_proba(feats)
    return alpha * zs + (1 - alpha) * pc
````

### `python/train.py`

````python
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
````

### `python/predict.py`

````python
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
````

### `python/app.py`

````python
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
````

### `python/requirements.txt`

````text
torch
transformers>=4.40,<5
pillow
numpy
scikit-learn
joblib
tqdm
gradio>=4
````

(`python/breeds.py`는 웹의 `BREEDS` 배열과 동일한 `(영문, 한글)` 목록 71개이며 `BREEDS = [...]`, `KO = dict(BREEDS)` 형태입니다. 웹 코드의 배열에서 그대로 생성하면 됩니다.)

## 7-3. 단일 HTML 빌더

### `embed/build_embedded.py`

````python
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
````

## 7-4. `embed/template.html` = `web/index.html` + 아래 변경

변경 요지: ① 외부 import 제거 → 내장 파일 풀기·fetch 가로채기·blob URL 동적 import, ② `loadModels`를 이미지 모델 먼저·`dtype:"q8", device:"wasm"`로 변경, ③ `</main>` 뒤에 `<!--EMBEDDED_DATA-->` 자리표시자 추가(빌드 시 gzip+base64 `<script class="emb" data-name="...">`들로 치환), ④ 안내 문구 수정.

````diff
@@ -51,7 +51,7 @@
   </label>
   <input id="file" type="file" accept="image/*" hidden>
 
-  <div id="status">모델을 준비하는 중입니다. 처음 한 번만 약 200MB를 내려받습니다.</div>
+  <div id="status">내장된 모델을 준비하는 중입니다. 파일이 커서 열리는 데 시간이 걸릴 수 있습니다.</div>
   <progress id="prog" max="100" value="0"></progress>
 
   <section id="result" aria-live="polite">
@@ -76,9 +76,48 @@
   </details>
 </main>
 
+<!--EMBEDDED_DATA-->
+
 <script type="module">
-import { AutoTokenizer, AutoProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, RawImage }
-  from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1";
+// ---------- 내장 파일 풀기 (인터넷 없이 동작) ----------
+const $s = document.getElementById("status");
+function b64ToU8(b64) {
+  const len = b64.length, pad = b64.endsWith("==") ? 2 : b64.endsWith("=") ? 1 : 0;
+  const out = new Uint8Array(len / 4 * 3 - pad); let o = 0; const CH = 4 * 1024 * 1024;
+  for (let i = 0; i < len; i += CH) { const bin = atob(b64.slice(i, i + CH)); for (let j = 0; j < bin.length; j++) out[o++] = bin.charCodeAt(j); }
+  return out;
+}
+const files = {};
+try {
+  const els = [...document.querySelectorAll("script.emb")];
+  for (let k = 0; k < els.length; k++) {
+    $s.textContent = `내장 파일을 푸는 중입니다… (${k + 1}/${els.length})`;
+    await new Promise(r => setTimeout(r));
+    const raw = b64ToU8(els[k].textContent.trim()); els[k].textContent = "";
+    const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip"));
+    files[els[k].dataset.name] = new Uint8Array(await new Response(stream).arrayBuffer());
+  }
+} catch (e) { $s.textContent = "내장 파일을 풀지 못했습니다: " + e.message; throw e; }
+
+const realFetch = window.fetch.bind(window);
+window.fetch = async (input, init) => {
+  const u = typeof input === "string" ? input : (input.url || String(input));
+  const m = u.match(/\/resolve\/[^/]+\/(.+?)(\?.*)?$/);
+  if (m) {
+    const f = files["hf/" + decodeURIComponent(m[1])];
+    if (!f) return new Response("not found", { status: 404 });
+    return new Response(f, { status: 200, headers: { "Content-Type": m[1].endsWith(".json") ? "application/json" : "application/octet-stream", "Content-Length": String(f.length) } });
+  }
+  return realFetch(input, init);
+};
+const blobUrl = (name, type) => URL.createObjectURL(new Blob([files[name]], { type }));
+const T = await import(blobUrl("lib.js", "text/javascript"));
+const { AutoTokenizer, AutoProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, RawImage, env } = T;
+env.allowLocalModels = false;
+env.useBrowserCache = false;
+env.backends.onnx.wasm.wasmPaths = { mjs: blobUrl("ort.mjs", "text/javascript"), wasm: blobUrl("ort.wasm", "application/wasm") };
+env.backends.onnx.wasm.numThreads = 1;
+env.backends.onnx.wasm.proxy = false;
 
 const MODEL = "Xenova/clip-vit-base-patch32";
 const MAX_PER_BREED = 40;   // 품종당 최대 기준 사진 수
@@ -127,14 +166,13 @@
 // ---------- 모델 ----------
 let tokenizer, textModel, processor, visionModel;
 async function loadModels() {
-  const cb = p => { if (p.status === "progress") setProg(p.progress); };
-  status("모델을 내려받는 중입니다 (텍스트)…");
-  tokenizer = await AutoTokenizer.from_pretrained(MODEL);
-  textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL, { progress_callback: cb });
-  status("모델을 내려받는 중입니다 (이미지)…");
+  const opt = { dtype: "q8", device: "wasm" };
+  status("모델을 불러오는 중입니다 (이미지)…");
   processor = await AutoProcessor.from_pretrained(MODEL);
-  visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL, { progress_callback: cb });
-  setProg(null);
+  visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL, opt);
+  status("모델을 불러오는 중입니다 (텍스트)…");
+  tokenizer = await AutoTokenizer.from_pretrained(MODEL);
+  textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL, opt);
 }
 async function embedTexts(texts) {
   const out = [];
@@ -294,7 +332,7 @@
   dis(false);
 } catch (e) {
   setProg(null);
-  status("모델을 불러오지 못했습니다. 인터넷 연결을 확인하고 새로고침하세요. (" + e.message + ")");
+  status("모델을 불러오지 못했습니다. 크롬 또는 엣지 최신 버전으로 다시 열어 보세요. (" + e.message + ")");
 }
 
 $("file").addEventListener("change", e => run(e.target.files[0]));
````
