"""Shared test helpers."""
import tkinter

import pytest


def require_display():
    """Skip the calling test when no usable desktop session exists.

    開發機的螢幕被鎖住／無頭環境下，tkinter 的 Tk() 會拋 TclError
    （「tk wasn't installed properly」的假象）。這種情況不是程式問題，
    跳過而不是失敗，讓本機打包不受螢幕鎖定狀態影響。
    """
    try:
        root = tkinter.Tk()
        root.destroy()
    except tkinter.TclError:
        pytest.skip("no available desktop session for tkinter")
