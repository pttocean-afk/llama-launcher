"""DPI 縮放工具：讓固定像素值跟著 Windows 顯示縮放（125%／150%…）等比放大。

背景：Tk 對 point 單位的字型會依 DPI 自動放大（`tk scaling`），但寫死的
像素值——視窗 geometry、minsize、固定 height/width、PanedWindow 分割——
不會跟著縮放。高 DPI 螢幕上字變大、容器沒變大，內容就會被切掉。

用法：
    DpiScale.init(root)          # 任何視窗建立前，每個 process 一次
    height=S(58)                 # 設計基準（100% 縮放）的 58px
    fit_window_size(root, S(1180), S(820))   # 再 clamp 進螢幕範圍

在無顯示環境（headless 測試）或讀不到 DPI 時一律安全退回 1.0。
"""
from __future__ import annotations


class DpiScale:
    """process 級縮放係數（96 DPI = 1.0）。"""

    factor: float = 1.0

    @classmethod
    def init(cls, root) -> float:
        """以 root 所在螢幕的 DPI 更新係數；回傳更新後的值。"""
        try:
            dpi = float(root.winfo_fpixels("1i"))
        except Exception:
            dpi = 96.0
        if not dpi > 0 or dpi != dpi:  # 非正數或 NaN 防禦
            dpi = 96.0
        # 下限 0.5：允許低 DPI（72dpi X11）等比縮小；上限 3.0：擋掉異常回報
        cls.factor = max(0.5, min(dpi / 96.0, 3.0))
        return cls.factor


def S(px: int) -> int:
    """把設計基準（100% 縮放）的像素值換算成目前 DPI 下的像素值。"""
    return max(1, int(round(px * DpiScale.factor)))


def fit_window_size(root, width: int, height: int,
                    screen_ratio: float = 0.9) -> tuple[int, int]:
    """把（已乘上 DPI 係數的）視窗尺寸 clamp 進主螢幕範圍。

    低解析度筆電（如 1366x768）不會因為縮放後的預設尺寸而裝不下。
    螢幕資訊取得失敗時原樣返回。"""
    try:
        screen_w = int(root.winfo_screenwidth())
        screen_h = int(root.winfo_screenheight())
    except Exception:
        return width, height
    if screen_w <= 0 or screen_h <= 0:
        return width, height
    return (max(320, min(width, int(screen_w * screen_ratio))),
            max(240, min(height, int(screen_h * screen_ratio))))
