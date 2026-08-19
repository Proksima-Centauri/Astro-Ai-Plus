#!/usr/bin/env python3
"""
Quick update notifier helper.

Use cases:
1) Publish update metadata to your API (admin side).
2) Check for updates at app startup and show Update / Do it later messagebox.

API contract (example):
- POST {api_url}/updates
  body: {
    "app_id": "astro-ai-plus",
    "target_os": "linux",
    "version": "1.4.0",
    "changes": ["New star shrink", "3D FLY fixes"],
    "update_url": "https://..."
  }

- GET {api_url}/updates/latest?app_id=astro-ai-plus&target_os=linux
  response: {
    "version": "1.4.0",
    "changes": ["New star shrink", "3D FLY fixes"],
    "update_url": "https://..."
  }

- POST {api_url}/updates/decision (optional)
  body: {
    "app_id": "astro-ai-plus",
    "target_os": "linux",
    "current_version": "1.3.0",
    "offered_version": "1.4.0",
    "decision": "update" | "later"
  }
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:  # pragma: no cover
    tk = None
    messagebox = None


def normalize_os(value: str | None = None) -> str:
    raw = (value or platform.system()).strip().lower()
    if raw in {"linux", "linux2"}:
        return "linux"
    if raw in {"windows", "win32", "cygwin"}:
        return "windows"
    if raw in {"darwin", "mac", "macos", "osx"}:
        return "macos"
    return raw


def parse_version(version: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", version)
    if not nums:
        return (0,)
    return tuple(int(n) for n in nums)


def is_newer(latest: str, current: str) -> bool:
    return parse_version(latest) > parse_version(current)


def _http_json(url: str, method: str = "GET", payload: dict | None = None, token: str | None = None) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url=url, method=method, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
        if not raw.strip():
            return {}
        return json.loads(raw)


def publish_update(api_url: str, app_id: str, target_os: str, version: str, changes: list[str], update_url: str, token: str | None = None) -> dict:
    payload = {
        "app_id": app_id,
        "target_os": normalize_os(target_os),
        "version": version,
        "changes": changes,
        "update_url": update_url,
    }
    return _http_json(f"{api_url.rstrip('/')}/updates", method="POST", payload=payload, token=token)


def fetch_latest_update(api_url: str, app_id: str, target_os: str, token: str | None = None) -> dict:
    query = urllib.parse.urlencode({"app_id": app_id, "target_os": normalize_os(target_os)})
    return _http_json(f"{api_url.rstrip('/')}/updates/latest?{query}", method="GET", token=token)


def send_decision(api_url: str, app_id: str, target_os: str, current_version: str, offered_version: str, decision: str, token: str | None = None) -> None:
    payload = {
        "app_id": app_id,
        "target_os": normalize_os(target_os),
        "current_version": current_version,
        "offered_version": offered_version,
        "decision": decision,
    }
    _http_json(f"{api_url.rstrip('/')}/updates/decision", method="POST", payload=payload, token=token)


def _build_message(version: str, changes: list[str], update_url: str) -> str:
    lines = [f"Nowa wersja: {version}", "", "Zmiany:"]
    for item in changes[:8]:
        lines.append(f"- {item}")
    if update_url:
        lines += ["", f"Link: {update_url}"]
    lines += ["", "Kliknij Yes aby pobrac aktualizacje."]
    return "\n".join(lines)


def ask_update_now(version: str, changes: list[str], update_url: str) -> str:
    if tk is None or messagebox is None:
        return "later"

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    text = _build_message(version, changes, update_url)
    do_update = messagebox.askyesno("Aktualizacja dostepna", text)
    root.destroy()
    return "update" if do_update else "later"


def check_for_updates_on_start(api_url: str, app_id: str, current_version: str, target_os: str | None = None, token: str | None = None) -> bool:
    os_name = normalize_os(target_os)
    try:
        latest = fetch_latest_update(api_url=api_url, app_id=app_id, target_os=os_name, token=token)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False

    latest_version = str(latest.get("version", "")).strip()
    if not latest_version or not is_newer(latest_version, current_version):
        return False

    changes = latest.get("changes") or []
    if not isinstance(changes, list):
        changes = [str(changes)]
    update_url = str(latest.get("update_url", "")).strip()

    decision = ask_update_now(version=latest_version, changes=[str(c) for c in changes], update_url=update_url)

    try:
        send_decision(
            api_url=api_url,
            app_id=app_id,
            target_os=os_name,
            current_version=current_version,
            offered_version=latest_version,
            decision=decision,
            token=token,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        pass

    if decision == "update" and update_url:
        webbrowser.open(update_url)
        return True
    return False


def run_simple_gui() -> int:
    if tk is None or messagebox is None:
        print("Tkinter not available.")
        return 1

    root = tk.Tk()
    root.title("Update Notifier")
    root.geometry("760x640")

    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack(fill="both", expand=True)

    api_var = tk.StringVar(value="https://twoje-api.pl")
    app_var = tk.StringVar(value="astro-ai-plus")
    os_var = tk.StringVar(value=normalize_os())
    version_var = tk.StringVar(value="1.0.0")
    current_var = tk.StringVar(value="1.0.0")
    url_var = tk.StringVar(value="https://example.com/download")
    token_var = tk.StringVar(value="")

    def add_row(row: int, label: str, var: tk.StringVar) -> None:
        tk.Label(frame, text=label, anchor="w").grid(row=row, column=0, sticky="w", pady=4)
        tk.Entry(frame, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)

    frame.columnconfigure(1, weight=1)
    add_row(0, "API URL", api_var)
    add_row(1, "App ID", app_var)

    tk.Label(frame, text="Target OS", anchor="w").grid(row=2, column=0, sticky="w", pady=4)
    tk.OptionMenu(frame, os_var, "linux", "windows", "macos").grid(row=2, column=1, sticky="w", pady=4)

    add_row(3, "New version", version_var)
    add_row(4, "Current version", current_var)
    add_row(5, "Update URL", url_var)
    add_row(6, "Token (optional)", token_var)

    tk.Label(frame, text="Changes (one per line)", anchor="w").grid(row=7, column=0, sticky="nw", pady=4)
    changes_box = tk.Text(frame, height=8)
    changes_box.grid(row=7, column=1, sticky="nsew", pady=4)
    frame.rowconfigure(7, weight=1)

    tk.Label(frame, text="Result", anchor="w").grid(row=8, column=0, sticky="nw", pady=4)
    result_box = tk.Text(frame, height=10)
    result_box.grid(row=8, column=1, sticky="nsew", pady=4)
    frame.rowconfigure(8, weight=1)

    def set_result(text: str) -> None:
        result_box.delete("1.0", "end")
        result_box.insert("1.0", text)

    def on_publish() -> None:
        changes = [line.strip() for line in changes_box.get("1.0", "end").splitlines() if line.strip()]
        try:
            result = publish_update(
                api_url=api_var.get().strip(),
                app_id=app_var.get().strip(),
                target_os=os_var.get().strip(),
                version=version_var.get().strip(),
                changes=changes,
                update_url=url_var.get().strip(),
                token=token_var.get().strip() or None,
            )
            set_result(json.dumps(result, indent=2, ensure_ascii=True))
            messagebox.showinfo("OK", "Update metadata sent.")
        except Exception as exc:
            set_result(f"ERROR: {exc}")
            messagebox.showerror("Error", str(exc))

    def on_check() -> None:
        try:
            opened = check_for_updates_on_start(
                api_url=api_var.get().strip(),
                app_id=app_var.get().strip(),
                current_version=current_var.get().strip(),
                target_os=os_var.get().strip(),
                token=token_var.get().strip() or None,
            )
            set_result("UPDATE_OPENED" if opened else "NO_UPDATE")
        except Exception as exc:
            set_result(f"ERROR: {exc}")
            messagebox.showerror("Error", str(exc))

    btns = tk.Frame(frame)
    btns.grid(row=9, column=0, columnspan=2, sticky="w", pady=(8, 0))
    tk.Button(btns, text="Publish", command=on_publish).pack(side="left", padx=(0, 8))
    tk.Button(btns, text="Check update", command=on_check).pack(side="left")

    root.mainloop()
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Publish/check app updates via API JSON.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pub = sub.add_parser("publish", help="Publish update metadata")
    pub.add_argument("--api-url", required=True)
    pub.add_argument("--app-id", required=True)
    pub.add_argument("--target-os", required=True, choices=["linux", "windows", "macos"])
    pub.add_argument("--version", required=True)
    pub.add_argument("--change", action="append", default=[], help="Repeat for each change")
    pub.add_argument("--update-url", required=True)
    pub.add_argument("--token")

    chk = sub.add_parser("check", help="Check update and show messagebox")
    chk.add_argument("--api-url", required=True)
    chk.add_argument("--app-id", required=True)
    chk.add_argument("--current-version", required=True)
    chk.add_argument("--target-os", choices=["linux", "windows", "macos"])
    chk.add_argument("--token")

    sub.add_parser("gui", help="Open simple GUI")

    args = parser.parse_args(argv)

    if args.cmd == "publish":
        result = publish_update(
            api_url=args.api_url,
            app_id=args.app_id,
            target_os=args.target_os,
            version=args.version,
            changes=args.change,
            update_url=args.update_url,
            token=args.token,
        )
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0

    if args.cmd == "gui":
        return run_simple_gui()

    updated = check_for_updates_on_start(
        api_url=args.api_url,
        app_id=args.app_id,
        current_version=args.current_version,
        target_os=args.target_os,
        token=args.token,
    )
    print("UPDATE_OPENED" if updated else "NO_UPDATE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
