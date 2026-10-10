"""Wikimedia Commons에서 품종별 고양이 사진을 내려받아 dataset/<품종>/ 폴더를 만든다.

    python fetch_dataset.py --per-breed 100                # 전체 품종
    python fetch_dataset.py --per-breed 30 --only Persian Siamese

- 무료 라이선스(CC, 퍼블릭 도메인) 사진만 받고, 출처·라이선스를 dataset/attribution.csv에 기록한다.
- 품종 분류(Category)에 들어 있는 사진만 받는다. 분류가 없는 품종은 건너뛴다.
- 그래도 다른 고양이 사진이 섞일 수 있으니 학습 전에 폴더를 훑어보고 지우는 것을 권장한다.
- 표준 라이브러리만 사용한다. Wikimedia 정책에 맞춰 천천히 받는다.
"""
import argparse, csv, json, os, re, sys, time, urllib.parse, urllib.request

from breeds import BREEDS

API = "https://commons.wikimedia.org/w/api.php"
UA = "cat-breed-v2-dataset/1.0 (https://github.com/ocqhdlffj-lgtm/cat-breed-v2)"
FREE = re.compile(r"^(CC[ -]BY|CC0|Public domain|PD|GFDL)", re.I)
SKIP_TITLE = re.compile(r"(logo|map|flag|diagram|drawing|painting|stamp|icon|poster|chart|graph|statistic|registration|data|\.svg|\.pdf|\.tif)", re.I)
# 털 무늬·혼합 항목은 품종이 아니므로 제외
SKIP_BREEDS = {"domestic shorthair", "domestic longhair", "orange tabby", "tuxedo", "calico",
               "tortoiseshell", "silver tabby", "mackerel tabby"}


def get(url, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(5 * (i + 1)); continue
            raise
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError("요청 실패: " + url)


def api(**params):
    return json.loads(get(API + "?" + urllib.parse.urlencode({"action": "query", "format": "json", **params})))


def members(cat, limit=300):
    """분류 하나에 들어 있는 파일 제목과 하위 분류 제목."""
    files, subs, cont = [], [], {}
    while len(files) < limit:
        d = api(list="categorymembers", cmtitle=cat, cmtype="file|subcat", cmlimit=200, **cont)
        for m in d.get("query", {}).get("categorymembers", []):
            (subs if m["title"].startswith("Category:") else files).append(m["title"])
        if "continue" not in d:
            break
        cont = d["continue"]; time.sleep(1)
    return files, subs


def info(titles):
    out = []
    for i in range(0, len(titles), 40):
        d = api(titles="|".join(titles[i:i + 40]), prop="imageinfo", iiprop="url|mime|extmetadata", iiurlwidth=640)
        for p in d.get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [{}])[0]
            meta = ii.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "")
            if ii.get("mime") not in ("image/jpeg", "image/png") or not FREE.match(lic):
                continue
            if SKIP_TITLE.search(p["title"]) or not ii.get("thumburl"):
                continue
            artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
            out.append((p["title"], ii["thumburl"], lic, artist, ii.get("descriptionurl", "")))
        time.sleep(1)
    return out


def search(en, want):
    """품종 분류(Category)에서만 사진을 찾는다. 글자 검색은 엉뚱한 사진이 섞여서 쓰지 않는다."""
    for cat in (f"Category:{en} cats", f"Category:{en} (cat)", f"Category:{en}"):
        files, subs = members(cat)
        if files or subs:
            break
    else:
        return []
    for sub in subs[:6]:           # 하위 분류(새끼 고양이 등) 한 단계만
        if len(files) >= want * 2:
            break
        files += members(sub, 100)[0]
    return info(files[: want * 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--per-breed", type=int, default=100)
    ap.add_argument("--only", nargs="*", help="이 영문 품종명만 받기")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, "attribution.csv")
    new_log = not os.path.exists(log_path)
    log = open(log_path, "a", newline="", encoding="utf-8")
    w = csv.writer(log)
    if new_log:
        w.writerow(["breed", "file", "title", "license", "author", "page"])

    for en, ko in BREEDS:
        if en in SKIP_BREEDS or (a.only and en not in a.only):
            continue
        folder = os.path.join(a.out, en)
        os.makedirs(folder, exist_ok=True)
        have = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))])
        if have >= a.per_breed:
            print(f"{en}: 이미 {have}장"); continue
        try:
            hits = search(en, a.per_breed - have)
        except Exception as e:
            print(f"{en}: 검색 실패 ({e})"); continue
        if not hits:
            print(f"{en}: Wikimedia에 품종 분류가 없어 건너뜀"); continue
        n = have
        for title, url, lic, artist, page in hits:
            if n >= a.per_breed:
                break
            name = re.sub(r"[^\w.-]+", "_", title.replace("File:", ""))[:80]
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                name += ".jpg"
            path = os.path.join(folder, name)
            if os.path.exists(path):
                continue
            try:
                data = get(url)
            except Exception:
                continue
            with open(path, "wb") as f:
                f.write(data)
            w.writerow([en, name, title, lic, artist, page]); log.flush()
            n += 1
            time.sleep(0.5)
        print(f"{en}: {n}장")
    log.close()


if __name__ == "__main__":
    main()
