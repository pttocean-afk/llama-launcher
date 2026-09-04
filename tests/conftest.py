"""Shared test helpers."""
import tkinter

import pytest


def require_tk(tk_module=tkinter):
    """建立測試實際使用的 Tk root；桌面不可用時直接 skip。

    不先建立探測用 root 再建立第二個，避免 Windows 在兩步之間鎖屏時
    產生競態（第一個成功、第二個拋 TclError）。
    """
    try:
        return tk_module.Tk()
    except tk_module.TclError:
        pytest.skip("no available desktop session for tkinter")
