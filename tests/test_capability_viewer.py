import time
import tkinter as tk
from pathlib import Path

from llama_launcher.capability_analysis import aggregate_capabilities
from llama_launcher.capability_viewer import (
    CapabilityViewer,
    format_context,
)
from llama_launcher.log_analysis import parse_log_file

FIXTURES = Path(__file__).parent / "fixtures" / "logs"


from conftest import require_tk

def test_context_label():
    assert format_context(131072) == "128K"
    assert format_context(200000) == "200K"
    assert format_context(None) == "?"


def test_capability_viewer_scans_and_renders():
    root = require_tk(tk)
    root.withdraw()
    viewer = CapabilityViewer(root, FIXTURES)
    deadline = time.monotonic() + 5
    try:
        while viewer.report is None and time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)
        assert viewer.report is not None
        assert viewer.runtime_var.get() in {
            "BeeLlama v0.4.4", "b10621", "b10509"
        }
        assert viewer.report.rows
        assert viewer.model.parsed
        assert viewer.tabs.index("end") == 3
        assert viewer.envelope_table.get_children()
    finally:
        viewer.destroy()
        root.destroy()
