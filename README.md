# Llama Launcher

Cross-platform desktop control center for local `llama.cpp` servers on Windows and Linux. It provides model profiles, process adoption, logs, a token-protected control page, and optional Tailscale Serve HTTPS remote access.

## Current status

This repository is being productized from a proven Windows launcher. The existing daily-use launcher remains separate until migration and platform validation are complete.

Verified so far:

- shared configuration, token, Tailscale, and diagnostics modules
- Windows x64 Portable build and Setup EXE
- Windows artifact startup and adoption of an existing `llama-server.exe --port 8080`
- Linux x86_64 PyInstaller bundle startup under Xvfb
- Linux control page responding on `127.0.0.1:8765`
- automated unit tests on Windows and Linux build workflows

A real Linux desktop tray session and real Linux `llama-server` launch still require hardware-host verification before v1.0.

## Platform behavior

| Area | Windows | Linux |
|---|---|---|
| llama.cpp executable | `llama-server.exe` | `llama-server` |
| User data | `%LOCALAPPDATA%\LlamaLauncher` | `$XDG_DATA_HOME/LlamaLauncher` or `~/.local/share/LlamaLauncher` |
| Process discovery | `psutil` | `psutil` |
| Remote access | Tailscale Serve HTTPS | Tailscale Serve HTTPS |
| Close button default | Minimize to tray | Exit safely; tray support varies by desktop |
| Release artifact | Setup EXE + Portable ZIP | Portable x86_64 tar.gz |

Runtime paths are host-local settings. Importing profiles never assumes that a Windows path is valid on Linux or vice versa.

## First-run flow

1. Select the folder containing `llama-server.exe` or `llama-server`.
2. Keep GGUF files in its `models` subfolder or configure profiles from the UI.
3. Press **Remote Access** to detect Tailscale and configure Serve.
4. If Tailscale requires one-time authorization, the app opens the authorization page.
5. Copy the generated HTTPS URL and private control token.

The control API always binds to `127.0.0.1:8765`. Tailscale Serve is the secure remote layer. The llama.cpp inference port `8080` is not exposed as the launcher-control channel.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

Windows build:

```powershell
.\scripts\build-windows.ps1
.\scripts\build-installer.ps1
```

Linux build (use the distribution system Python with Tk installed):

```bash
sudo apt-get install python3-tk python3-venv xvfb
python3 -m venv .venv-build
PATH="$PWD/.venv-build/bin:$PATH" bash scripts/build-linux.sh
```

## Private data excluded from Git

- control tokens
- GGUF models and mmproj files
- machine-specific runtime/model paths
- logs and backups
- Tailscale hostnames
- generated build artifacts
