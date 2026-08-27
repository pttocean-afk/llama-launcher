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

## Verified

- Linux unit suite: 10 passed.
- Python source and tests compile on Linux.
- Windows prototype artifact previously passed 8 tests and adopted the running Vulkan server without restarting it; it must be rebuilt after the cross-platform process-layer change.
- Linux system-Python artifact starts under Xvfb and serves the dashboard with HTTP 200.
- First Linux build made with Hermes Python failed due missing `libtcl9.0.so`; build procedure was corrected to distribution system Python plus `python3-tk`.

## Remaining before v1.0

- Rebuild and re-smoke-test the Windows artifacts after cross-platform refactor.
- Validate a real native Linux `llama-server` start/stop/adopt cycle on a Linux GPU host.
- Validate tray behavior in an actual GNOME/KDE desktop session; Xvfb has no tray manager.
- Replace the hard-coded Windows VRAM preflight with optional host-policy configuration.
- Complete safe legacy settings migration UI and profile export/import.
- Add Linux startup integration and decide whether AppImage/deb packaging is warranted.
- Perform migration rehearsal before replacing the existing daily-use launcher.
- Create GitHub repository and publish only after owner review.
