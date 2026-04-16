"""Grayscale viewport filter plugin for EOV (Python gRPC).

This plugin connects to the EOV extension host via gRPC, registers a
named remote plugin session, contributes a viewport HUD toolbar button, and
processes CPU filter frames over the shared Python helper in plugin_api/python.
"""

from pathlib import Path
import sys
import threading

try:
    import grpc
except ImportError:
    print("[grayscale_python] grpcio not installed; gRPC filter unavailable.", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    np = None

try:
    from eov_plugin_host import ExtensionHostClient
except ImportError:
    helper_candidates = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent / "python_api",
        Path(__file__).resolve().parents[2] / "plugin_api" / "python",
    ]
    for helper_root in helper_candidates:
        helper_file = helper_root / "eov_plugin_host.py"
        descriptor_file = helper_root / "eov_extension.desc"
        if helper_file.exists() and descriptor_file.exists():
            sys.path.insert(0, str(helper_root))
            break
    from eov_plugin_host import ExtensionHostClient


HUD_ICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="9"/>
    <path d="M8 10h.01"/>
    <path d="M16 10h.01"/>
    <path d="M8 15c1.2-1 2.6-1.5 4-1.5s2.8.5 4 1.5"/>
</svg>
""".strip()

def _apply_grayscale(rgba_bytes, width, height):
    """Convert RGBA bytes to grayscale in-place (luminance) and return bytes."""
    length = width * height * 4
    if np is not None:
        arr = np.frombuffer(rgba_bytes, dtype=np.uint8).copy().reshape(-1, 4)
        lum = (0.299 * arr[:, 0] + 0.587 * arr[:, 1] + 0.114 * arr[:, 2]).astype(np.uint8)
        arr[:, 0] = lum
        arr[:, 1] = lum
        arr[:, 2] = lum
        return arr.tobytes()
    else:
        buf = bytearray(rgba_bytes)
        for i in range(0, length, 4):
            r, g, b = buf[i], buf[i + 1], buf[i + 2]
            lum = int(0.299 * r + 0.587 * g + 0.114 * b)
            buf[i] = lum
            buf[i + 1] = lum
            buf[i + 2] = lum
        return bytes(buf)

def main():
    plugin_id = "grayscale_python"
    button_id = "toggle_grayscale_hud"
    action_id = "toggle_grayscale_hud"
    enabled = False

    action_thread_running = False
    client = None
    filter_stream = None
    session = None

    try:
        client = ExtensionHostClient.from_env()
        session = client.register_plugin(
            plugin_id=plugin_id,
            display_name="Grayscale (Python)",
            version="0.1.0",
            language="python",
        )
        snapshot = session.initial_snapshot
        active_filename = snapshot.active_file.filename if snapshot.has_active_file else ""
        print(
            f"[grayscale_python] Connected to {snapshot.app_name} {snapshot.app_version}; active file={active_filename or '<none>'}"
        )

        session.log_message(
            "info",
            f"Connected to {snapshot.app_name} {snapshot.app_version} (active file: {active_filename or 'none'})",
        )

        session.register_hud_toolbar_button(
            button_id=button_id,
            tooltip="Toggle Grayscale In This Viewport (Python)",
            icon_svg=HUD_ICON_SVG,
            action_id=action_id,
        )
        print("[grayscale_python] HUD toolbar button registered")

        filter_id = session.register_filter(
            name="Grayscale (Python)",
            supports_cpu=True,
            supports_gpu=False,
        )
        print(f"[grayscale_python] Registered filter with id={filter_id}")

        session.set_filter_enabled(filter_id, False)
        print("[grayscale_python] Filter disabled by default")

        filter_stream = session.open_cpu_filter_stream(filter_id)
        toolbar_actions = session.hud_toolbar_action_stream()
        action_thread_running = True

        def action_loop():
            nonlocal enabled, action_thread_running
            try:
                for action in toolbar_actions:
                    if action.plugin_id != plugin_id or action.action_id != action_id:
                        continue
                    enabled = not enabled
                    session.set_filter_enabled(filter_id, enabled)
                    state = "enabled" if enabled else "disabled"
                    pane = action.viewport.pane_index if action.viewport is not None else -1
                    print(
                        f"[grayscale_python] HUD action received: pane={pane} {action.button_id} -> {state}"
                    )
                    session.log_message(
                        "info",
                        f"Grayscale filter {state} from HUD pane {pane}",
                    )
            except grpc.RpcError as err:
                if action_thread_running:
                    print(f"[grayscale_python] HUD toolbar stream error: {err}", file=sys.stderr)

        action_thread = threading.Thread(target=action_loop, daemon=True)
        action_thread.start()

        print("[grayscale_python] Stream connected, processing frames...")
        for frame in filter_stream:
            if frame.width == 0 or frame.height == 0 or not frame.rgba_data:
                continue
            processed = _apply_grayscale(frame.rgba_data, frame.width, frame.height)
            filter_stream.send_processed_frame(frame.width, frame.height, processed)
    except grpc.RpcError as err:
        print(f"[grayscale_python] gRPC error: {err}", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        action_thread_running = False
        if filter_stream is not None:
            filter_stream.close()
        if session is not None:
            try:
                session.unregister()
                print(f"[grayscale_python] Unregistered plugin session {session.plugin_handle}")
            except grpc.RpcError:
                pass
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
