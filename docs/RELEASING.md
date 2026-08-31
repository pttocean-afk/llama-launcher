# 打包與發布 SOP（Windows + Linux）

本專案每次改版都發布**雙平台**（Windows x64 + Linux x86_64）。本篇是完整流程，跨 session 直接照抄，不要再重新摸索工具鏈。

**流程總覽**

```
改完程式 → 本地測試過 → README 更新（截圖/徽章）→
  Windows 打包（docs/BUILD-WINDOWS.md）→ Linux 打包（本篇 §2）→
  GitHub Release 上傳（本篇 §3）→ 驗證下載
```

---

## 0. 前置檢查（每次必做）

```bash
cd /home/pttocean/projects/llama-launcher

# 完整測試（tkinter 測試需要顯示伺服器，Linux 用 xvfb）
xvfb-run -a .venv-build/bin/python -m pytest -q        # 預期全過（目前 122）
.venv-build/bin/python -m compileall -q src tests      # 語法檢查

# 確認 git 乾淨、README 測試徽章數字與 pytest 一致（見 §4）
git status --short
```

---

## 1. Windows 打包

**不要重新找方法**：完整 SOP 在 [docs/BUILD-WINDOWS.md](BUILD-WINDOWS.md)（WSL→Windows interop、PyInstaller、Inno Setup 路徑、踩坑清單）。

核心指令（詳見該文件）：

```bash
# WSL 內
rsync -a --exclude '.git' --exclude '.venv*' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude 'build' --exclude 'dist' --exclude '*.pyc' \
  ./ /mnt/e/llama-launcher-build/

cd /mnt/e/llama-launcher-build
/mnt/c/Windows/py.exe -m PyInstaller --noconfirm --clean --windowed \
  --name LlamaLauncher --icon "src/llama_launcher/assets/llama-launcher-icon.ico" \
  --add-data "src/llama_launcher/assets;assets" --paths src scripts/launcher_entry.py

cp README.md dist/LlamaLauncher/README.md
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
  "Compress-Archive -Path 'E:\llama-launcher-build\dist\LlamaLauncher\*' -DestinationPath 'E:\llama-launcher-build\dist\LlamaLauncher-Portable-x64.zip' -Force"

cd installer
"/mnt/c/Users/pttoc/AppData/Local/Programs/Inno Setup 6/ISCC.exe" LlamaLauncher.iss

# 複製回 repo
cd /home/pttocean/projects/llama-launcher
cp -f /mnt/e/llama-launcher-build/dist/LlamaLauncher-Setup-x64.exe dist/
cp -f /mnt/e/llama-launcher-build/dist/LlamaLauncher-Portable-x64.zip dist/
```

產出：`dist/LlamaLauncher-Setup-x64.exe`、`dist/LlamaLauncher-Portable-x64.zip`

---

## 2. Linux 打包

用 repo 內的 `.venv-build`（**系統 Python，不要用 Hermes 內建 Python**；需有 Tk）：

```bash
cd /home/pttocean/projects/llama-launcher

# 第一次才需要建 venv：
#   sudo apt-get install python3-tk python3-venv xvfb
#   python3 -m venv .venv-build

PATH="$PWD/.venv-build/bin:$PATH" xvfb-run -a bash scripts/build-linux.sh
```

`scripts/build-linux.sh` 自己會：`pip install -e '.[dev]'` → `pytest` → PyInstaller（`--add-data` 用 `:` 分隔、PNG icon）→ 複製 README 與 desktop file → 打包 tar.gz。

> ⚠️ 腳本內的 pytest 需要顯示伺服器，**一定要包 `xvfb-run -a`**，否則 tkinter 測試會失敗。

產出：`dist/LlamaLauncher-Linux-x86_64.tar.gz`（解壓後為 `LlamaLauncher/` 資料夾）

---

## 3. GitHub Release 上傳

用 git 憑證（`credential.helper=store` 已存好 `pttocean-afk` 的 token），透過 `python3` ＋ `urllib` 打 GitHub API，**不需要使用者再給 key**。

> 需要時機：新版本首次發佈（`vX.Y.Z`），或舊 release 補傳資產。重複發佈同一 tag 會失敗，改版就換新 tag。

```bash
cd /home/pttocean/projects/llama-launcher

cat > /tmp/make_release.py <<'PYEOF'
import json, subprocess, urllib.request, urllib.error, sys

VERSION = "v0.1.0"                      # ← 每次改版本號
NAME = "Llama Launcher v0.1.0"          # ← release 顯示名稱
BODY = ("Windows x64 安裝檔與可攜版，Linux x86_64 tar.gz。\n\n"
        "Windows installer + portable, Linux x86_64 tarball.")

def get_cred():
    out = subprocess.run(["git", "credential", "fill"],
                         input="protocol=https\nhost=github.com\n\n",
                         capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            d[k] = v
    return d.get("username"), d.get("password")

user, token = get_cred()
assert token, "No stored git credential — run: git credential-store 或 gh auth login"

REPO = "pttocean-afk/llama-launcher"
BASE = f"https://api.github.com/repos/{REPO}"

def api(path, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "llama-launcher-release-script")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")

status, rel = api("/releases", "POST", {
    "tag_name": VERSION, "name": NAME, "body": BODY,
    "draft": False, "prerelease": False,
})
print("create release:", status, rel.get("id") or rel.get("message"))
assert status in (200, 201), "release create failed"
rid = rel["id"]

for path, name in [
    ("dist/LlamaLauncher-Setup-x64.exe", "LlamaLauncher-Setup-x64.exe"),
    ("dist/LlamaLauncher-Portable-x64.zip", "LlamaLauncher-Portable-x64.zip"),
    ("dist/LlamaLauncher-Linux-x86_64.tar.gz", "LlamaLauncher-Linux-x86_64.tar.gz"),
]:
    blob = open(path, "rb").read()
    url = f"https://uploads.github.com/repos/{REPO}/releases/{rid}/assets?name={name}"
    req = urllib.request.Request(url, data=blob, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "llama-launcher-release-script")
    req.add_header("Content-Type", "application/octet-stream")
    try:
        with urllib.request.urlopen(req) as r:
            a = json.loads(r.read().decode())
            print("upload", name, "OK")
    except urllib.error.HTTPError as e:
        print("upload", name, "FAILED", e.code, e.read().decode()[:300]); sys.exit(1)
print("RELEASE_DONE")
PYEOF

.venv-build/bin/python /tmp/make_release.py
```

**只補傳資產到已存在的 release**（同 tag 不重建）：

```bash
# 取得 release id
curl -sL https://api.github.com/repos/pttocean-afk/llama-launcher/releases/tags/vX.Y.Z \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])"
# 用上一個腳本但把 create 段改為 GET /releases/tags/<tag>，其餘上傳段相同
```

**沒有 API 的備案**：網頁 https://github.com/pttocean-afk/llama-launcher/releases/new → 填 tag → 拖曳三個檔案上傳。

---

## 4. README 更新（發佈前檢查）

- **測試徽章**：第 9 行 `tests-122%20passed-green`，數字要跟著 `pytest` 結果更新（目前 122）。
- **截圖**：`docs/screenshot.png`（主畫面）、`docs/screenshot-settings.png`（設定分頁）需反映最新 UI。
  - 重新截圖方法（無頭環境）：`xvfb-run -a -s "-screen 0 1280x900x24" .venv-build/bin/python`，用 PIL `ImageGrab.grab()` 抓視窗。請參考本 repo 曾用過的截圖腳本（`/tmp/capture_final.py` 樣式），關鍵點：
    - `app_mod.HAVE_TRAY = False`（Xvfb 無系統匣，pystray 會卡）
    - 在 `models/` 放假的 `.gguf`/`llama-server` 檔案，避免 `run_first_setup` 彈出模態 `askdirectory` 卡死
    - `LLAMA_SERVER` 必須指向**真實存在的檔案**，否則會進 first-setup
- README 雙語（中/英）都要同步改。

改完 README → `git add` → commit → push（README / docs 變更都在 repo 內）。

---

## 5. 發佈後驗證

```bash
for u in \
  https://github.com/pttocean-afk/llama-launcher/releases/download/vX.Y.Z/LlamaLauncher-Setup-x64.exe \
  https://github.com/pttocean-afk/llama-launcher/releases/download/vX.Y.Z/LlamaLauncher-Portable-x64.zip \
  https://github.com/pttocean-afk/llama-launcher/releases/download/vX.Y.Z/LlamaLauncher-Linux-x86_64.tar.gz ; do
  curl -sL -o /dev/null -w "%{http_code} $u\n" "$u"
done
# 全部應回 200；角色權限不足時可能回 302，需再 -L 追蹤
```

---

## 6. 踩坑備忘

| 坑 | 解法 |
|---|---|
| 忘了版本號、重複建同 tag | GitHub API 回 `422`；改新 tag 或改用「補傳資產」方式 |
| Linux pytest 掛在 tkinter | 一定要 `xvfb-run -a` 包住整個 build/測試 |
| `.venv-build` 指向 Hermes Python | Linux build 要用**系統 python3**（`which python3` 檢查）；Hermes venv 沒有 Tk |
| Windows `--add-data` 用 `:` | Windows 上分隔符是 `;`（`assets;assets`），Linux 才是 `:` |
| git credential 不存在 | `git config credential.helper store` 後 push 一次存下，或用 `gh auth login` |
| 改版時 README 徽章/截圖過時 | 見 §4；截圖流程已標準化，別手動截 |
| dist/ 是 gitignore | 打包產物不上 git，只上 GitHub Releases |