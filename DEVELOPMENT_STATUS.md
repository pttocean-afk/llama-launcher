# Development Status

## Product boundary

Llama Launcher must run natively on both Windows and Linux because the managed `llama.cpp` runtime may live on either host. Windows-only launch or management paths are not acceptable as the shared core.

## Completed foundation

- Created clean Git repository separate from the existing daily-use Windows launcher.
- Moved settings, profiles, logs, and token to per-user data directories.
- Added atomic JSON writes and stable private token generation.
- Added Tailscale CLI discovery, Serve setup, authorization URL detection, and diagnostics.
- Added first-run llama.cpp directory selection.
- Added exact `--port 8080` process adoption through a shared `psutil` host layer.
- Added cross-platform process-tree termination and external file/URL opening.
- Added OS-specific `llama-server` executable naming and XDG/LOCALAPPDATA paths.
- Kept the existing Windows dual-GPU VRAM cleanup policy Windows-only; Linux does not execute Windows/WSL cleanup commands.
- Added Windows Portable ZIP and per-user Inno Setup installer definitions.
- Added Linux x86_64 portable tar.gz build and desktop entry.
- Added Windows and Linux GitHub Actions build/release workflows.

## Security fixes (2026-08-27, DSH)

- Remote dashboard XSS fixed: profile rows are now built with `createElement`/`textContent`
  and `addEventListener` (no `innerHTML` interpolation, no model path in `onclick`).
- Added a shared `threading.Lock` around all start/stop operations: Tk `on_launch`/`on_stop`,
  degraded auto-stop, quit-time stop, and remote `remote_start`/`remote_stop` all serialize
  on the same lock, so the HTTP thread and the Tk thread can no longer start/stop the
  managed server concurrently.
- Control API bind failure is no longer silent: the bind error is logged to the
  `llama_launcher` logger, exposed in `/api/status` as `control.ok`/`control.error`,
  and shown in a startup warning dialog.
- New test suite `tests/test_app_security.py`: DOM-render assertions, lock behavior of
  remote start/stop, bind-failure observability (requires a display; run under xvfb-run),
  and a real HTTP round-trip through `ControlServer` (auth 401 + token + profile start).

## Legacy migration (2026-08-27, DSH)

- `migration.py` extended: `detect_legacy_dir`, `plan_migration` (dry-run preview),
  `merge_legacy_into_config` (in-place idempotent merge, new-app settings win),
  `save_cfg` (atomic write). `migrate_legacy_data` now merges profiles into an existing
  config instead of skipping the file.
- `app.py`: `on_migrate_legacy` / `_do_migrate_legacy` / `_maybe_offer_legacy_migration`
  + "Import old launcher data" button. Auto-offers on first run when no profiles exist
  and a legacy folder is detected (checks `llama_dir` setting, `~/llama-cpp/launcher-app`,
  `C:\llama-cpp\launcher-app`, etc.).
- Tests: `tests/test_migration_ui.py` (8 tests: detection, planning, fresh + existing
  destination, merge semantics, idempotency, full app flow).

## Verified

- Linux unit suite: 23 passed under Xvfb; 22 headless (1 display-bound test).
- Python source and tests compile on Linux.
- Windows x64 Portable ZIP and per-user Setup EXE rebuilt after the cross-platform process-layer change; Windows test suite: 10 passed.
- Rebuilt Windows artifact started successfully, served Tailscale HTTPS with HTTP 200, and adopted the live Vulkan llama-server PID 22092 without restart.
- Linux system-Python artifact starts under Xvfb and serves the dashboard with HTTP 200.
- First Linux build made with Hermes Python failed due missing `libtcl9.0.so`; build procedure was corrected to distribution system Python plus `python3-tk`.
- Linux artifact rebuilt after the security fixes: starts under Xvfb, dashboard HTTP 200,
  `/api/status` reports `control.ok: true`, unauthenticated `/api/status` returns 401,
  and the served page uses the DOM-based profile renderer (no `innerHTML` profile data).
- Windows Portable ZIP rebuilt via WSL→Windows Python 3.14 + PyInstaller 6.18
  (repo copied to `E:\llama-launcher-build`, build output copied back to `dist/`).
  `LlamaLauncher-Setup-x64.exe` (Inno Setup) not rebuilt — Inno Setup not found on
  this machine; the existing EXE is from the pre-security-fix build.

## Remaining before v1.0

- Validate a real native Linux `llama-server` start/stop/adopt cycle on a Linux GPU host.
- Validate tray behavior in an actual GNOME/KDE desktop session; Xvfb has no tray manager.
- Replace the hard-coded Windows VRAM preflight with optional host-policy configuration.
- ~~Complete safe legacy settings migration UI~~ ✅ done (commit c4d034e).
- Profile export/import (without local absolute paths).
- Add Linux startup integration and decide whether AppImage/deb packaging is warranted.
- Perform migration rehearsal before replacing the existing daily-use launcher.
- Create GitHub repository and publish only after owner review.
