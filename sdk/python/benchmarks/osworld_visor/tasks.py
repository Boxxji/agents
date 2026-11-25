"""10 real multi-step OS/computer-workflow tasks modeled on OSWorld + external JSON loader.

All 10 built-in tasks are clean-room Apache-2.0 multi-step workflows spanning
OSWorld's desktop and computer-use domains:
1. ``osw_01_filesystem_batch_ops`` (os)
2. ``osw_02_spreadsheet_csv_transform`` (libreoffice_calc)
3. ``osw_03_sqlite_data_pipeline`` (multi_apps)
4. ``osw_04_cli_git_automation`` (vs_code)
5. ``osw_05_config_and_archive_ops`` (os)
6. ``osw_06_document_markdown_report`` (libreoffice_writer)
7. ``osw_07_local_http_service_check`` (chrome)
8. ``osw_08_permissions_and_cleanup_audit`` (os)
9. ``osw_09_log_triage_incident_report`` (multi_apps)
10. ``osw_10_json_yaml_pipeline_patch`` (vs_code)

Every task executes real OS operations inside an isolated subprocess sandbox and
is graded by an independent evaluator inspecting real disk/database/git/archive
state. Upstream ``xlang-ai/OSWorld`` Apache-2.0 JSON task manifests can also be
loaded at runtime via ``load_external_osworld_dir()``.
"""

from __future__ import annotations

import configparser
import csv
import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schemas import (
    DesktopToolingReport,
    ExternalOSWorldTask,
    OSWorldTaskSpec,
    TaskCategory,
    WorkflowStep,
)

# Domains in upstream xlang-ai/OSWorld that require a graphical X11/VM desktop.
GUI_ONLY_OSWORLD_DOMAINS = frozenset(
    {
        "chrome",
        "gimp",
        "libreoffice_calc",
        "libreoffice_impress",
        "libreoffice_writer",
        "thunderbird",
        "vlc",
    }
)


def _py_step(name: str, description: str, code: str, *, working_subdir: str = "", timeout_s: float = 20.0) -> WorkflowStep:
    return WorkflowStep(
        name=name,
        description=description,
        command=[sys.executable, "-c", code.strip()],
        timeout_s=timeout_s,
        working_subdir=working_subdir,
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Task 1: Filesystem batch organization, cleanup & SHA-256 manifest
# ---------------------------------------------------------------------------

def _task_01_filesystem_batch_ops() -> OSWorldTaskSpec:
    seed_files = {
        "inbox/app.log": "2026-11-01 INFO boot ok\n2026-11-01 INFO ready\n",
        "inbox/worker.log": "2026-11-01 WARN retry=1\n",
        "inbox/readme.txt": "Project Handbook   \nLine two\t  \nLine three\n",
        "inbox/notes.txt": "Meeting notes   \nFollow up  \n",
        "inbox/metrics.csv": "id,val\n1,10\n2,20\n",
        "inbox/users.csv": "uid,role\nu1,admin\nu2,viewer\n",
        "inbox/scratch.tmp": "temporary garbage\n",
        "inbox/old.bak": "obsolete backup\n",
        "inbox/.env.sample": "APP_ENV=staging\nLOG_LEVEL=info\n",
    }
    step1 = """
import pathlib, shutil
inbox = pathlib.Path("inbox")
org = pathlib.Path("organized")
for sub in ("logs", "docs", "data"):
    (org / sub).mkdir(parents=True, exist_ok=True)
for item in sorted(inbox.iterdir()):
    if item.name.endswith((".tmp", ".bak")):
        item.unlink()
    elif item.suffix == ".log":
        shutil.move(str(item), str(org / "logs" / item.name))
    elif item.suffix == ".txt":
        lines = [ln.rstrip(" \\t") for ln in item.read_text(encoding="utf-8").splitlines()]
        (org / "docs" / item.name).write_text("\\n".join(lines) + "\\n", encoding="utf-8")
        item.unlink()
    elif item.suffix == ".csv":
        shutil.move(str(item), str(org / "data" / item.name))
"""
    step2 = """
import hashlib, json, pathlib
org = pathlib.Path("organized")
entries = {}
for fp in sorted(org.rglob("*")):
    if fp.is_file():
        raw = fp.read_bytes()
        rel = fp.relative_to(org).as_posix()
        entries[rel] = {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
pathlib.Path("manifest.json").write_text(json.dumps({"files": entries, "count": len(entries)}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
"""
    return OSWorldTaskSpec(
        task_id="osw_01_filesystem_batch_ops",
        title="Batch filesystem organization, sanitization & SHA-256 manifest",
        category=TaskCategory.OS_FILESYSTEM,
        osworld_domain="os",
        instruction=(
            "Organize inbox/ into organized/{logs,docs,data}/ by file extension, "
            "strip trailing whitespace from .txt files, delete .tmp and .bak files, "
            "preserve dotfiles (.env.sample) in inbox/, and emit manifest.json with "
            "exact byte sizes and SHA-256 checksums."
        ),
        seed_files=seed_files,
        oracle_steps=[
            _py_step("organize_and_clean", "Move files by type, strip .txt trailing whitespace, remove .tmp/.bak", step1),
            _py_step("emit_manifest", "Compute deterministic SHA-256 manifest.json", step2),
        ],
        evaluator_kind="filesystem_batch_ops",
    )


# ---------------------------------------------------------------------------
# Task 2: Spreadsheet/CSV cleaning, deduplication & department tax rollups
# ---------------------------------------------------------------------------

def _task_02_spreadsheet_csv_transform() -> OSWorldTaskSpec:
    expenses_csv = (
        "txn_id,department,amount,updated_at\n"
        "T102,Engineering,\" $1,250.50 \",2026-10-01T10:00:00Z\n"
        "T101,Marketing,$400.00,2026-10-01T08:00:00Z\n"
        "T102,Engineering,\"$1,300.00\",2026-10-02T09:00:00Z\n"
        "T103,Engineering,-$100.00,2026-10-03T11:00:00Z\n"
        "T104,Operations,\"$2,500.25\",2026-10-03T12:00:00Z\n"
        "T101,Marketing,$450.00,2026-10-04T08:30:00Z\n"
        "T105,Operations,$500.75,2026-10-05T14:00:00Z\n"
    )
    tax_rates = json.dumps(
        {"Engineering": 0.10, "Marketing": 0.08, "Operations": 0.05},
        indent=2,
    ) + "\n"
    step1 = """
import csv, json, pathlib
raw_rows = list(csv.DictReader(pathlib.Path("finance/q3_expenses.csv").open(encoding="utf-8")))
latest = {}
for row in raw_rows:
    tid = row["txn_id"].strip()
    amt_str = row["amount"].strip().replace("$", "").replace(",", "")
    rec = {
        "txn_id": tid,
        "department": row["department"].strip(),
        "amount": round(float(amt_str), 2),
        "updated_at": row["updated_at"].strip(),
    }
    if tid not in latest or rec["updated_at"] > latest[tid]["updated_at"]:
        latest[tid] = rec
ordered = [latest[k] for k in sorted(latest)]
with pathlib.Path("finance/q3_clean.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["txn_id", "department", "amount", "updated_at"])
    w.writeheader()
    for rec in ordered:
        w.writerow({**rec, "amount": f"{rec['amount']:.2f}"})
"""
    step2 = """
import csv, json, pathlib
rates = json.loads(pathlib.Path("finance/tax_rates.json").read_text(encoding="utf-8"))
totals = {}
with pathlib.Path("finance/q3_clean.csv").open(encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        dept = row["department"]
        totals[dept] = round(totals.get(dept, 0.0) + float(row["amount"]), 2)
summary = {}
for dept in sorted(totals):
    net = round(totals[dept], 2)
    tax = round(net * rates[dept], 2)
    summary[dept] = {"net_spend": net, "tax_amount": tax, "gross_total": round(net + tax, 2)}
pathlib.Path("finance/department_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
"""
    return OSWorldTaskSpec(
        task_id="osw_02_spreadsheet_csv_transform",
        title="CSV expense deduplication, currency normalization & tax summary",
        category=TaskCategory.SPREADSHEET_DATA,
        osworld_domain="libreoffice_calc",
        instruction=(
            "Deduplicate finance/q3_expenses.csv by txn_id keeping the latest updated_at, "
            "normalize currency values, write sorted finance/q3_clean.csv, and compute "
            "per-department net_spend, tax_amount, and gross_total in finance/department_summary.json."
        ),
        seed_files={
            "finance/q3_expenses.csv": expenses_csv,
            "finance/tax_rates.json": tax_rates,
        },
        oracle_steps=[
            _py_step("clean_csv", "Deduplicate transactions and normalize currency formatting", step1),
            _py_step("compute_rollups", "Compute per-department tax rollups JSON", step2),
        ],
        evaluator_kind="spreadsheet_csv_transform",
    )


# ---------------------------------------------------------------------------
# Task 3: SQLite telemetry ingestion, dead-letter quarantine & SLA view
# ---------------------------------------------------------------------------

def _task_03_sqlite_data_pipeline() -> OSWorldTaskSpec:
    schema_sql = (
        "CREATE TABLE IF NOT EXISTS events (\n"
        "  event_id TEXT PRIMARY KEY,\n"
        "  service TEXT NOT NULL,\n"
        "  status INTEGER NOT NULL,\n"
        "  latency_ms REAL NOT NULL,\n"
        "  ts TEXT NOT NULL\n"
        ");\n"
    )
    events = [
        {"event_id": "e01", "service": "auth", "status": 200, "latency_ms": 12.0, "ts": "2026-11-01T00:01:00Z"},
        {"event_id": "e02", "service": "auth", "status": 200, "latency_ms": 18.0, "ts": "2026-11-01T00:02:00Z"},
        {"event_id": "e03", "service": "auth", "status": 500, "latency_ms": 95.0, "ts": "2026-11-01T00:03:00Z"},
        {"event_id": "e04", "service": "auth", "status": 200, "latency_ms": 22.0, "ts": "2026-11-01T00:04:00Z"},
        {"event_id": "e05", "service": "billing", "status": 200, "latency_ms": 40.0, "ts": "2026-11-01T00:05:00Z"},
        {"event_id": "e06", "service": "billing", "status": 503, "latency_ms": 210.0, "ts": "2026-11-01T00:06:00Z"},
        {"event_id": "e07", "service": "billing", "status": 200, "latency_ms": 55.0, "ts": "2026-11-01T00:07:00Z"},
        {"event_id": "e08", "service": "billing", "status": 502, "latency_ms": 180.0, "ts": "2026-11-01T00:08:00Z"},
        {"event_id": "e09", "service": "search", "status": 200, "latency_ms": 8.5, "ts": "2026-11-01T00:09:00Z"},
        {"event_id": "e10", "service": "search", "status": 200, "latency_ms": 11.5, "ts": "2026-11-01T00:10:00Z"},
        {"event_id": "e11", "service": "search", "status": 200, "latency_ms": 15.0, "ts": "2026-11-01T00:11:00Z"},
        {"event_id": "e12", "service": "search", "status": 200, "latency_ms": 14.0, "ts": "2026-11-01T00:12:00Z"},
    ]
    raw_lines = [json.dumps(e) for e in events[:4]]
    raw_lines.append("{malformed_json_line_one")
    raw_lines.extend(json.dumps(e) for e in events[4:8])
    raw_lines.append('{"event_id": "bad_missing_fields", "service": "auth"}')
    raw_lines.extend(json.dumps(e) for e in events[8:])
    raw_lines.append("not-json-at-all")
    events_jsonl = "\n".join(raw_lines) + "\n"

    step1 = """
import json, pathlib, sqlite3
conn = sqlite3.connect("db/analytics.sqlite")
conn.executescript(pathlib.Path("db/schema.sql").read_text(encoding="utf-8"))
dead = []
required = {"event_id", "service", "status", "latency_ms", "ts"}
for line in pathlib.Path("db/events_raw.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        obj = json.loads(line)
        if not isinstance(obj, dict) or not required.issubset(obj):
            raise ValueError("missing fields")
        conn.execute(
            "INSERT OR REPLACE INTO events(event_id, service, status, latency_ms, ts) VALUES (?, ?, ?, ?, ?)",
            (str(obj["event_id"]), str(obj["service"]), int(obj["status"]), float(obj["latency_ms"]), str(obj["ts"])),
        )
    except Exception:
        dead.append(line)
conn.execute("CREATE INDEX IF NOT EXISTS idx_events_service ON events(service)")
conn.execute(
    "CREATE VIEW IF NOT EXISTS service_sla AS "
    "SELECT service, COUNT(*) AS total_requests, "
    "SUM(CASE WHEN status >= 500 THEN 1 ELSE 0 END) AS error_count, "
    "ROUND(AVG(latency_ms), 2) AS avg_latency_ms, "
    "MAX(latency_ms) AS max_latency_ms "
    "FROM events GROUP BY service ORDER BY service"
)
conn.commit()
conn.close()
pathlib.Path("db/dead_letter.jsonl").write_text("\\n".join(dead) + ("\\n" if dead else ""), encoding="utf-8")
"""
    step2 = """
import json, pathlib, sqlite3
conn = sqlite3.connect("db/analytics.sqlite")
rows = conn.execute("SELECT service, total_requests, error_count, avg_latency_ms, max_latency_ms FROM service_sla").fetchall()
conn.close()
report = {
    r[0]: {
        "total_requests": r[1],
        "error_count": r[2],
        "error_rate": round(r[2] / r[1], 4),
        "avg_latency_ms": r[3],
        "max_latency_ms": r[4],
    }
    for r in rows
}
pathlib.Path("db/sla_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
"""
    return OSWorldTaskSpec(
        task_id="osw_03_sqlite_data_pipeline",
        title="SQLite JSONL ingestion, dead-letter quarantine & SLA view export",
        category=TaskCategory.DATABASE_PIPELINE,
        osworld_domain="multi_apps",
        instruction=(
            "Create db/analytics.sqlite from db/schema.sql, ingest valid JSONL rows from "
            "db/events_raw.jsonl, quarantine malformed rows to db/dead_letter.jsonl, create "
            "index idx_events_service and view service_sla, and export db/sla_report.json."
        ),
        seed_files={
            "db/schema.sql": schema_sql,
            "db/events_raw.jsonl": events_jsonl,
        },
        oracle_steps=[
            _py_step("ingest_sqlite", "Initialize DB, ingest valid rows, quarantine dead letters, create index & view", step1),
            _py_step("export_sla", "Query service_sla view and write db/sla_report.json", step2),
        ],
        evaluator_kind="sqlite_data_pipeline",
    )


# ---------------------------------------------------------------------------
# Task 4: Real Git release automation (branch, cherry-pick, changelog, tag)
# ---------------------------------------------------------------------------

def _task_04_cli_git_automation() -> OSWorldTaskSpec:
    git_spec = {
        "subdir": "repo",
        "base_files": {
            "VERSION": "1.1.0\n",
            "CHANGELOG.md": "# Changelog\n\n## 1.1.0\n- Initial release\n",
            "calc.py": (
                "def compute_ratio(num: float, den: float) -> float:\n"
                "    return num / den\n"
            ),
        },
        "base_commit_msg": "chore: release 1.1.0",
        "feature_branch": "feature/safe-ratio",
        "feature_files": {
            "calc.py": (
                "def compute_ratio(num: float, den: float) -> float:\n"
                "    if den == 0:\n"
                "        return 0.0\n"
                "    return round(num / den, 4)\n"
            ),
        },
        "feature_commit_msg": "fix: guard zero denominator in compute_ratio",
        "untracked_files": {
            "scratch_debug.tmp": "do not commit me\n",
        },
    }
    step1 = """
import pathlib, subprocess
def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()
scratch = pathlib.Path("scratch_debug.tmp")
if scratch.exists():
    scratch.unlink()
feature_sha = git("rev-parse", "feature/safe-ratio")
short_sha = git("rev-parse", "--short", feature_sha)
git("checkout", "-b", "release/v1.2")
git("cherry-pick", feature_sha)
pathlib.Path("VERSION").write_text("1.2.0\\n", encoding="utf-8")
cl = pathlib.Path("CHANGELOG.md").read_text(encoding="utf-8")
entry = f"\\n## 1.2.0\\n- Guard zero denominator in compute_ratio ({short_sha})\\n"
pathlib.Path("CHANGELOG.md").write_text(cl + entry, encoding="utf-8")
git("add", "VERSION", "CHANGELOG.md")
git("commit", "-m", "chore(release): prepare v1.2.0")
git("tag", "v1.2.0")
"""
    return OSWorldTaskSpec(
        task_id="osw_04_cli_git_automation",
        title="Git release branch creation, cherry-pick, changelog & tag",
        category=TaskCategory.DEV_CLI_GIT,
        osworld_domain="vs_code",
        instruction=(
            "Inside repo/, remove scratch_debug.tmp, create branch release/v1.2 from main, "
            "cherry-pick the fix commit from feature/safe-ratio, bump VERSION to 1.2.0, "
            "append a 1.2.0 entry referencing the short SHA to CHANGELOG.md, commit cleanly, "
            "and tag v1.2.0."
        ),
        seed_files={},
        setup_git_repo=git_spec,
        oracle_steps=[
            _py_step("git_release", "Branch, cherry-pick, update VERSION & CHANGELOG, commit & tag", step1, working_subdir="repo"),
        ],
        evaluator_kind="cli_git_automation",
    )


# ---------------------------------------------------------------------------
# Task 5: Config merge, secret redaction, gzipped tar bundle & SHA-256 verify
# ---------------------------------------------------------------------------

def _task_05_config_and_archive_ops() -> OSWorldTaskSpec:
    app_ini = (
        "[server]\n"
        "host = 127.0.0.1\n"
        "port = 8080\n"
        "workers = 2\n"
        "tls = false\n\n"
        "[logging]\n"
        "level = debug\n"
        "format = text\n"
    )
    overrides = json.dumps(
        {
            "server": {"port": "8443", "workers": "8", "tls": "true"},
            "logging": {"level": "info"},
        },
        indent=2,
    ) + "\n"
    secrets_tpl = (
        "DB_HOST=db.internal\n"
        "DB_PASSWORD=placeholder\n"
        "API_TOKEN=placeholder\n"
        "APP_REGION=eu-west-1\n"
    )
    step1 = """
import configparser, json, pathlib
cfg = configparser.ConfigParser()
cfg.read("deploy/app.ini", encoding="utf-8")
overrides = json.loads(pathlib.Path("deploy/overrides.json").read_text(encoding="utf-8"))
for sec, kvs in overrides.items():
    if not cfg.has_section(sec):
        cfg.add_section(sec)
    for k, v in kvs.items():
        cfg.set(sec, k, str(v))
with pathlib.Path("deploy/app.prod.ini").open("w", encoding="utf-8") as fh:
    cfg.write(fh)
redacted_lines = []
for line in pathlib.Path("deploy/secrets.template").read_text(encoding="utf-8").splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        if any(tok in k.upper() for tok in ("PASSWORD", "TOKEN", "SECRET")):
            v = "[REDACTED]"
        redacted_lines.append(f"{k}={v}")
pathlib.Path("deploy/redacted.env").write_text("\\n".join(redacted_lines) + "\\n", encoding="utf-8")
"""
    step2 = """
import hashlib, pathlib, tarfile
dist = pathlib.Path("dist")
dist.mkdir(parents=True, exist_ok=True)
bundle = dist / "release-bundle.tar.gz"
with tarfile.open(bundle, "w:gz") as tf:
    tf.add("deploy/app.prod.ini", arcname="app.prod.ini")
    tf.add("deploy/redacted.env", arcname="redacted.env")
    for fp in sorted(pathlib.Path("deploy/assets").rglob("*")):
        if fp.is_file():
            tf.add(fp, arcname="assets/" + fp.relative_to("deploy/assets").as_posix())
digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
(dist / "release-bundle.sha256").write_text(f"{digest}  release-bundle.tar.gz\\n", encoding="utf-8")
"""
    return OSWorldTaskSpec(
        task_id="osw_05_config_and_archive_ops",
        title="INI config merge, secret redaction & verified tar.gz release bundle",
        category=TaskCategory.ARCHIVE_CONFIG,
        osworld_domain="os",
        instruction=(
            "Merge deploy/overrides.json into deploy/app.ini -> deploy/app.prod.ini, "
            "redact PASSWORD/TOKEN/SECRET values from deploy/secrets.template -> deploy/redacted.env, "
            "archive app.prod.ini, redacted.env, and deploy/assets/ into dist/release-bundle.tar.gz, "
            "and write its SHA-256 digest to dist/release-bundle.sha256."
        ),
        seed_files={
            "deploy/app.ini": app_ini,
            "deploy/overrides.json": overrides,
            "deploy/secrets.template": secrets_tpl,
            "deploy/assets/banner.txt": "Hanzo Visor Production Bundle\n",
            "deploy/assets/routes.json": '{"health": "/healthz"}\n',
        },
        oracle_steps=[
            _py_step("merge_and_redact", "Merge INI overrides and redact sensitive env keys", step1),
            _py_step("pack_and_hash", "Create release-bundle.tar.gz and release-bundle.sha256", step2),
        ],
        evaluator_kind="config_and_archive_ops",
    )


# ---------------------------------------------------------------------------
# Task 6: Document synthesis (Markdown executive report + action items JSON)
# ---------------------------------------------------------------------------

def _task_06_document_markdown_report() -> OSWorldTaskSpec:
    meeting_a = (
        "# Architecture Sync\n"
        "- [ ] @alice Finish Visor snapshot GC audit (due: 2026-11-28)\n"
        "- [x] @bob Verify S3 multipart retries (due: 2026-11-20)\n"
        "- [ ] @carol Benchmark cold-start p95 latency (due: 2026-11-26)\n"
    )
    meeting_b = (
        "# Release Readiness\n"
        "- [ ] @alice Publish HIP-1348 report tables (due: 2026-11-27)\n"
        "- [x] @carol Pin OSWorld Apache-2.0 license audit (due: 2026-11-22)\n"
    )
    metrics_json = json.dumps(
        {
            "cold_start_ms": {"baseline": 120.0, "current": 42.0},
            "success_rate_pct": {"baseline": 80.0, "current": 100.0},
            "leaked_orphans": {"baseline": 3, "current": 0},
        },
        indent=2,
    ) + "\n"
    step1 = """
import json, pathlib, re
pat = re.compile(r"^- \\[( |x)\\] (@\\w+) (.+?) \\(due: (\\d{4}-\\d{2}-\\d{2})\\)$")
open_items, done_items = [], []
for fn in ("notes/meeting_a.md", "notes/meeting_b.md"):
    for line in pathlib.Path(fn).read_text(encoding="utf-8").splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        rec = {"owner": m.group(2), "task": m.group(3), "due": m.group(4)}
        (done_items if m.group(1) == "x" else open_items).append(rec)
open_items.sort(key=lambda r: (r["owner"], r["due"], r["task"]))
done_items.sort(key=lambda r: (r["owner"], r["due"], r["task"]))
metrics = json.loads(pathlib.Path("notes/metrics.json").read_text(encoding="utf-8"))
reports = pathlib.Path("reports")
reports.mkdir(parents=True, exist_ok=True)
(reports / "action_items.json").write_text(json.dumps({"open": open_items, "completed": done_items}, indent=2) + "\\n", encoding="utf-8")
lines = [
    "# Weekly Executive Report",
    "",
    "## Executive Summary",
    f"Tracked {len(open_items)} open action items and {len(done_items)} completed items across engineering syncs.",
    "",
    "## Open Action Items",
]
for item in open_items:
    lines.append(f"- {item['owner']}: {item['task']} (due {item['due']})")
lines.extend(["", "## Completed Items"])
for item in done_items:
    lines.append(f"- {item['owner']}: {item['task']} (due {item['due']})")
lines.extend(["", "## KPI Table", "| Metric | Baseline | Current | Delta |", "| --- | ---: | ---: | ---: |"])
for k in sorted(metrics):
    b, c = metrics[k]["baseline"], metrics[k]["current"]
    d = round(c - b, 2)
    lines.append(f"| {k} | {b} | {c} | {d:+.2f} |")
(reports / "weekly_exec.md").write_text("\\n".join(lines) + "\\n", encoding="utf-8")
"""
    return OSWorldTaskSpec(
        task_id="osw_06_document_markdown_report",
        title="Executive Markdown report & action-item extraction from meeting notes",
        category=TaskCategory.DOCUMENT_OFFICE,
        osworld_domain="libreoffice_writer",
        instruction=(
            "Parse notes/meeting_a.md, notes/meeting_b.md, and notes/metrics.json to generate "
            "reports/weekly_exec.md (with Executive Summary, Open Action Items, Completed Items, "
            "and KPI Table with numeric deltas) plus reports/action_items.json."
        ),
        seed_files={
            "notes/meeting_a.md": meeting_a,
            "notes/meeting_b.md": meeting_b,
            "notes/metrics.json": metrics_json,
        },
        oracle_steps=[
            _py_step("build_report", "Extract action items, compute KPI deltas, write Markdown & JSON reports", step1),
        ],
        evaluator_kind="document_markdown_report",
    )


# ---------------------------------------------------------------------------
# Task 7: Local loopback HTTP server lifecycle, live GET probes & clean shutdown
# ---------------------------------------------------------------------------

def _task_07_local_http_service_check() -> OSWorldTaskSpec:
    index_html = "<!doctype html><html><body><h1>Visor Sandbox Ready</h1></body></html>\n"
    api_data = json.dumps({"service": "visor-local", "healthy": True, "version": 2}) + "\n"
    step1 = """
import functools, hashlib, http.server, json, pathlib, threading, time, urllib.request

static_dir = pathlib.Path("web/static").resolve()

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            payload = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()
    def log_message(self, format, *args):
        pass

handler_cls = functools.partial(Handler, directory=str(static_dir))
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
port = server.server_address[1]
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
results = {}
try:
    for route in ("/index.html", "/api_data.json", "/health"):
        t0 = time.perf_counter_ns()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{route}", timeout=5) as resp:
            body = resp.read()
            dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            results[route] = {
                "status": resp.status,
                "content_length": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "latency_ms": round(dt_ms, 4),
            }
finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)

report = {"port": port, "routes": results, "server_stopped": True}
pathlib.Path("web/probe_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
"""
    return OSWorldTaskSpec(
        task_id="osw_07_local_http_service_check",
        title="Loopback HTTP server probe & guaranteed port teardown",
        category=TaskCategory.WEB_LOCAL_HTTP,
        osworld_domain="chrome",
        instruction=(
            "Start an ephemeral loopback HTTP server on 127.0.0.1:0 serving web/static/ plus /health, "
            "probe /index.html, /api_data.json, and /health, record statuses, SHA-256 digests, and "
            "measured latencies in web/probe_report.json, then cleanly shut down the server."
        ),
        seed_files={
            "web/static/index.html": index_html,
            "web/static/api_data.json": api_data,
        },
        oracle_steps=[
            _py_step("serve_and_probe", "Launch loopback HTTP server, probe 3 endpoints, shut down cleanly", step1),
        ],
        evaluator_kind="local_http_service_check",
    )


# ---------------------------------------------------------------------------
# Task 8: POSIX permissions hardening, script execution & cache purge
# ---------------------------------------------------------------------------

def _task_08_permissions_and_cleanup_audit() -> OSWorldTaskSpec:
    run_sh = "#!/bin/sh\necho 'hardened-run-ok' > sec/output.txt\n"
    step1 = """
import json, os, pathlib, stat, subprocess
key_p = pathlib.Path("sec/keys/service.key")
script_p = pathlib.Path("sec/scripts/run.sh")
os.chmod(key_p, 0o600)
os.chmod(script_p, 0o755)
subprocess.run([str(script_p.resolve())], check=True)
removed = []
for fp in sorted(pathlib.Path("sec/cache").glob("*.tmp")):
    removed.append(fp.name)
    fp.unlink()
summary = {
    "key_mode": oct(stat.S_IMODE(key_p.stat().st_mode)),
    "script_mode": oct(stat.S_IMODE(script_p.stat().st_mode)),
    "removed_tmp_files": removed,
    "output_present": pathlib.Path("sec/output.txt").is_file(),
}
pathlib.Path("sec/audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
"""
    return OSWorldTaskSpec(
        task_id="osw_08_permissions_and_cleanup_audit",
        title="POSIX file permissions hardening, script execution & stale cache purge",
        category=TaskCategory.SYSTEM_SECURITY_CLEANUP,
        osworld_domain="os",
        instruction=(
            "Set sec/keys/service.key to 0o600, set sec/scripts/run.sh to 0o755, execute "
            "sec/scripts/run.sh to produce sec/output.txt, delete all *.tmp files in sec/cache/, "
            "and write sec/audit_summary.json."
        ),
        seed_files={
            "sec/keys/service.key": "PRIVATE-KEY-PLACEHOLDER\n",
            "sec/scripts/run.sh": run_sh,
            "sec/cache/stale_1.tmp": "stale1\n",
            "sec/cache/stale_2.tmp": "stale2\n",
            "sec/cache/keep.dat": "retain-me\n",
        },
        initial_modes={
            "sec/keys/service.key": 0o644,
            "sec/scripts/run.sh": 0o644,
        },
        oracle_steps=[
            _py_step("harden_and_clean", "Chmod key/script, run script, purge cache *.tmp, write audit JSON", step1),
        ],
        evaluator_kind="permissions_and_cleanup_audit",
    )


# ---------------------------------------------------------------------------
# Task 9: Multi-file log triage & incident correlation
# ---------------------------------------------------------------------------

def _task_09_log_triage_incident_report() -> OSWorldTaskSpec:
    log_1 = (
        "ts=2026-11-01T10:00:01Z level=INFO service=gateway code=200 req_id=r1\n"
        "ts=2026-11-01T10:00:05Z level=ERROR service=payments code=502 req_id=r2\n"
        "ts=2026-11-01T10:00:09Z level=ERROR service=payments code=502 req_id=r2\n"
    )
    log_2 = (
        "ts=2026-11-02T09:15:00Z level=WARN service=auth code=429 req_id=r3\n"
        "ts=2026-11-02T09:15:10Z level=CRITICAL service=payments code=503 req_id=r4\n"
        "ts=2026-11-02T09:16:00Z level=ERROR service=auth code=500 req_id=r5\n"
    )
    log_3 = (
        "ts=2026-11-03T12:00:00Z level=INFO service=payments code=200 req_id=r6\n"
        "ts=2026-11-03T12:01:00Z level=ERROR service=payments code=502 req_id=r2\n"
    )
    step1 = """
import collections, json, pathlib
error_lines = []
svc_5xx = collections.Counter()
code_counts = collections.Counter()
req_counts = collections.Counter()
for fp in sorted(pathlib.Path("logs").glob("*.log")):
    for line in fp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        kv = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
        level = kv.get("level", "")
        code = int(kv.get("code", "0"))
        if level in ("ERROR", "CRITICAL"):
            error_lines.append((kv.get("ts", ""), line))
            req_counts[kv.get("req_id", "")] += 1
        if code >= 500:
            svc_5xx[kv.get("service", "")] += 1
            code_counts[str(code)] += 1
error_lines.sort(key=lambda x: x[0])
triage = pathlib.Path("triage")
triage.mkdir(parents=True, exist_ok=True)
(triage / "error_lines.log").write_text("\\n".join(x[1] for x in error_lines) + "\\n", encoding="utf-8")
summary = {
    "total_error_lines": len(error_lines),
    "top_failing_service": svc_5xx.most_common(1)[0][0] if svc_5xx else "",
    "service_5xx_counts": dict(sorted(svc_5xx.items())),
    "code_5xx_counts": dict(sorted(code_counts.items())),
    "retried_error_req_ids": sorted(k for k, v in req_counts.items() if v > 1),
}
(triage / "incident_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
"""
    return OSWorldTaskSpec(
        task_id="osw_09_log_triage_incident_report",
        title="Multi-file rotated log triage & 5xx incident correlation",
        category=TaskCategory.OS_FILESYSTEM,
        osworld_domain="multi_apps",
        instruction=(
            "Parse logs/*.log, extract all ERROR/CRITICAL entries chronologically into "
            "triage/error_lines.log, and compute top_failing_service, service_5xx_counts, "
            "code_5xx_counts, and retried_error_req_ids into triage/incident_summary.json."
        ),
        seed_files={
            "logs/app-2026-11-01.log": log_1,
            "logs/app-2026-11-02.log": log_2,
            "logs/app-2026-11-03.log": log_3,
        },
        oracle_steps=[
            _py_step("triage_logs", "Parse rotated logs and emit triage summary + sorted error log", step1),
        ],
        evaluator_kind="log_triage_incident_report",
    )


# ---------------------------------------------------------------------------
# Task 10: CI pipeline & environment JSON migration + validator verification
# ---------------------------------------------------------------------------

def _task_10_json_yaml_pipeline_patch() -> OSWorldTaskSpec:
    workflow = json.dumps(
        {
            "name": "visor-ci",
            "jobs": {
                "lint": {"image": "hanzo/runner:v1-legacy", "steps": ["ruff check"]},
                "test": {
                    "image": "hanzo/runner:v1-legacy",
                    "timeout_minutes": 30,
                    "steps": ["pytest"],
                },
                "package": {"image": "hanzo/builder:v1-legacy", "steps": ["build"]},
            },
        },
        indent=2,
    ) + "\n"
    dev_env = json.dumps({"env": "dev", "image": "hanzo/app:v1-legacy", "replicas": 1}, indent=2) + "\n"
    staging_env = json.dumps({"env": "staging", "image": "hanzo/app:v1-legacy", "replicas": 2}, indent=2) + "\n"
    prod_env = json.dumps({"env": "prod", "image": "hanzo/app:v1-legacy", "replicas": 4, "read_only_root": False}, indent=2) + "\n"
    validator_py = (
        "import json, pathlib, sys\n"
        "wf = json.loads(pathlib.Path('ci/workflow.json').read_text())\n"
        "for name, job in wf['jobs'].items():\n"
        "    assert 'v1-legacy' not in job['image'], f'legacy tag in {name}'\n"
        "    assert 'timeout_minutes' in job and job['timeout_minutes'] > 0, f'missing timeout in {name}'\n"
        "for env_name in ('dev', 'staging', 'prod'):\n"
        "    cfg = json.loads(pathlib.Path(f'ci/environments/{env_name}.json').read_text())\n"
        "    assert 'v1-legacy' not in cfg['image'], f'legacy tag in {env_name}'\n"
        "    if env_name == 'prod':\n"
        "        assert cfg.get('read_only_root') is True, 'prod must enforce read_only_root'\n"
        "print('validator-ok')\n"
    )
    step1 = """
import json, pathlib
changes = {}
wf_p = pathlib.Path("ci/workflow.json")
wf = json.loads(wf_p.read_text(encoding="utf-8"))
wf_mods = []
for name, job in wf["jobs"].items():
    if "v1-legacy" in job["image"]:
        job["image"] = job["image"].replace("v1-legacy", "v2-hardened")
        wf_mods.append(f"jobs.{name}.image")
    if "timeout_minutes" not in job:
        job["timeout_minutes"] = 15
        wf_mods.append(f"jobs.{name}.timeout_minutes")
wf_p.write_text(json.dumps(wf, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
changes["ci/workflow.json"] = sorted(wf_mods)

for env_name in ("dev", "staging", "prod"):
    ep = pathlib.Path(f"ci/environments/{env_name}.json")
    cfg = json.loads(ep.read_text(encoding="utf-8"))
    mods = []
    if "v1-legacy" in cfg.get("image", ""):
        cfg["image"] = cfg["image"].replace("v1-legacy", "v2-hardened")
        mods.append("image")
    if env_name == "prod" and cfg.get("read_only_root") is not True:
        cfg["read_only_root"] = True
        mods.append("read_only_root")
    ep.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    changes[f"ci/environments/{env_name}.json"] = sorted(mods)

pathlib.Path("ci/migration_diff.json").write_text(json.dumps(changes, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
"""
    return OSWorldTaskSpec(
        task_id="osw_10_json_yaml_pipeline_patch",
        title="CI workflow & multi-environment JSON migration + schema validation",
        category=TaskCategory.DEV_CLI_GIT,
        osworld_domain="vs_code",
        instruction=(
            "Upgrade all v1-legacy image tags to v2-hardened across ci/workflow.json and "
            "ci/environments/{dev,staging,prod}.json, add timeout_minutes=15 to jobs lacking "
            "one, enforce read_only_root=true in prod.json, verify via ci/validator.py, and "
            "write ci/migration_diff.json."
        ),
        seed_files={
            "ci/workflow.json": workflow,
            "ci/environments/dev.json": dev_env,
            "ci/environments/staging.json": staging_env,
            "ci/environments/prod.json": prod_env,
            "ci/validator.py": validator_py,
        },
        oracle_steps=[
            _py_step("migrate_configs", "Patch CI workflow and environment manifests + emit migration diff", step1),
            WorkflowStep(
                name="run_validator",
                description="Run ci/validator.py inside isolated sandbox",
                command=[sys.executable, "ci/validator.py"],
                timeout_s=10.0,
            ),
        ],
        evaluator_kind="json_yaml_pipeline_patch",
    )


def build_osworld_subset() -> List[OSWorldTaskSpec]:
    """Return all 10 reproducible clean-room OSWorld-modeled computer-workflow tasks."""
    return [
        _task_01_filesystem_batch_ops(),
        _task_02_spreadsheet_csv_transform(),
        _task_03_sqlite_data_pipeline(),
        _task_04_cli_git_automation(),
        _task_05_config_and_archive_ops(),
        _task_06_document_markdown_report(),
        _task_07_local_http_service_check(),
        _task_08_permissions_and_cleanup_audit(),
        _task_09_log_triage_incident_report(),
        _task_10_json_yaml_pipeline_patch(),
    ]


# ---------------------------------------------------------------------------
# Independent Post-Execution Evaluators (run outside agent trust boundary)
# ---------------------------------------------------------------------------

def evaluate_task(
    task: OSWorldTaskSpec, workspace: Path
) -> Tuple[bool, float, Dict[str, bool], str]:
    """Independently verify real workspace state for a completed task."""
    evaluators = {
        "filesystem_batch_ops": _eval_01_filesystem_batch_ops,
        "spreadsheet_csv_transform": _eval_02_spreadsheet_csv_transform,
        "sqlite_data_pipeline": _eval_03_sqlite_data_pipeline,
        "cli_git_automation": _eval_04_cli_git_automation,
        "config_and_archive_ops": _eval_05_config_and_archive_ops,
        "document_markdown_report": _eval_06_document_markdown_report,
        "local_http_service_check": _eval_07_local_http_service_check,
        "permissions_and_cleanup_audit": _eval_08_permissions_and_cleanup_audit,
        "log_triage_incident_report": _eval_09_log_triage_incident_report,
        "json_yaml_pipeline_patch": _eval_10_json_yaml_pipeline_patch,
    }
    fn = evaluators.get(task.evaluator_kind)
    if fn is None:
        return False, 0.0, {"unknown_evaluator": False}, f"Unknown evaluator_kind={task.evaluator_kind}"
    checks = fn(workspace)
    passed_count = sum(1 for ok in checks.values() if ok)
    total = len(checks) or 1
    score = passed_count / total
    all_ok = passed_count == len(checks) and len(checks) > 0
    failed = [k for k, ok in checks.items() if not ok]
    details = "all_checks_passed" if all_ok else f"failed_checks={failed}"
    return all_ok, score, checks, details


def _eval_01_filesystem_batch_ops(ws: Path) -> Dict[str, bool]:
    org = ws / "organized"
    inbox = ws / "inbox"
    checks: Dict[str, bool] = {}
    checks["logs_moved"] = (org / "logs" / "app.log").is_file() and (org / "logs" / "worker.log").is_file()
    checks["data_moved"] = (org / "data" / "metrics.csv").is_file() and (org / "data" / "users.csv").is_file()
    readme_p = org / "docs" / "readme.txt"
    notes_p = org / "docs" / "notes.txt"
    checks["docs_trimmed"] = (
        readme_p.is_file()
        and notes_p.is_file()
        and readme_p.read_text(encoding="utf-8") == "Project Handbook\nLine two\nLine three\n"
        and notes_p.read_text(encoding="utf-8") == "Meeting notes\nFollow up\n"
    )
    checks["tmp_bak_deleted"] = not (inbox / "scratch.tmp").exists() and not (inbox / "old.bak").exists()
    checks["dotfile_preserved"] = (inbox / ".env.sample").is_file() and "APP_ENV=staging" in (inbox / ".env.sample").read_text(encoding="utf-8")
    manifest_p = ws / "manifest.json"
    manifest_ok = False
    if manifest_p.is_file():
        try:
            data = json.loads(manifest_p.read_text(encoding="utf-8"))
            files = data.get("files", {})
            if data.get("count") == 6 and len(files) == 6:
                manifest_ok = all(
                    (org / rel).is_file()
                    and info.get("sha256") == _sha256_file(org / rel)
                    and info.get("size") == (org / rel).stat().st_size
                    for rel, info in files.items()
                )
        except (OSError, ValueError, TypeError):
            manifest_ok = False
    checks["manifest_verified"] = manifest_ok
    return checks


def _eval_02_spreadsheet_csv_transform(ws: Path) -> Dict[str, bool]:
    clean_p = ws / "finance" / "q3_clean.csv"
    summary_p = ws / "finance" / "department_summary.json"
    checks: Dict[str, bool] = {}
    csv_ok = False
    if clean_p.is_file():
        try:
            rows = list(csv.DictReader(clean_p.open(encoding="utf-8")))
            ids = [r["txn_id"] for r in rows]
            by_id = {r["txn_id"]: float(r["amount"]) for r in rows}
            csv_ok = (
                ids == ["T101", "T102", "T103", "T104", "T105"]
                and by_id["T101"] == 450.0
                and by_id["T102"] == 1300.0
                and by_id["T103"] == -100.0
                and by_id["T104"] == 2500.25
                and by_id["T105"] == 500.75
            )
        except (OSError, ValueError, KeyError):
            csv_ok = False
    checks["clean_csv_exact"] = csv_ok

    summary_ok = False
    if summary_p.is_file():
        try:
            s = json.loads(summary_p.read_text(encoding="utf-8"))
            summary_ok = (
                s.get("Engineering") == {"net_spend": 1200.0, "tax_amount": 120.0, "gross_total": 1320.0}
                and s.get("Marketing") == {"net_spend": 450.0, "tax_amount": 36.0, "gross_total": 486.0}
                and s.get("Operations") == {"net_spend": 3001.0, "tax_amount": 150.05, "gross_total": 3151.05}
            )
        except (OSError, ValueError, TypeError):
            summary_ok = False
    checks["summary_json_exact"] = summary_ok
    return checks


def _eval_03_sqlite_data_pipeline(ws: Path) -> Dict[str, bool]:
    db_p = ws / "db" / "analytics.sqlite"
    dl_p = ws / "db" / "dead_letter.jsonl"
    rep_p = ws / "db" / "sla_report.json"
    checks: Dict[str, bool] = {}
    db_ok = False
    idx_ok = False
    view_ok = False
    if db_p.is_file():
        try:
            conn = sqlite3.connect(str(db_p))
            cnt = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            db_ok = cnt == 12
            idxs = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()]
            idx_ok = "idx_events_service" in idxs
            vrows = conn.execute("SELECT service, total_requests, error_count FROM service_sla ORDER BY service").fetchall()
            view_ok = vrows == [("auth", 4, 1), ("billing", 4, 2), ("search", 4, 0)]
            conn.close()
        except sqlite3.Error:
            pass
    checks["sqlite_rows_12"] = db_ok
    checks["index_created"] = idx_ok
    checks["view_sla_exact"] = view_ok
    checks["dead_letter_3"] = dl_p.is_file() and len([ln for ln in dl_p.read_text(encoding="utf-8").splitlines() if ln.strip()]) == 3
    rep_ok = False
    if rep_p.is_file():
        try:
            rep = json.loads(rep_p.read_text(encoding="utf-8"))
            rep_ok = (
                rep["auth"]["error_rate"] == 0.25
                and rep["billing"]["error_rate"] == 0.5
                and rep["search"]["error_rate"] == 0.0
                and rep["billing"]["max_latency_ms"] == 210.0
            )
        except (OSError, ValueError, KeyError, TypeError):
            rep_ok = False
    checks["sla_report_exact"] = rep_ok
    return checks


def _eval_04_cli_git_automation(ws: Path) -> Dict[str, bool]:
    repo = ws / "repo"
    checks: Dict[str, bool] = {}
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(ws),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
    }

    def _git(*args: str) -> str:
        res = subprocess.run(["git", *args], cwd=str(repo), env=env, capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else ""

    checks["branch_release_v1_2"] = _git("rev-parse", "--abbrev-ref", "HEAD") == "release/v1.2"
    checks["tag_v1_2_0_exists"] = "v1.2.0" in _git("tag", "-l", "v1.2.0").splitlines()
    checks["working_tree_clean"] = _git("status", "--porcelain") == "" and not (repo / "scratch_debug.tmp").exists()
    ver_p = repo / "VERSION"
    checks["version_1_2_0"] = ver_p.is_file() and ver_p.read_text(encoding="utf-8").strip() == "1.2.0"
    feat_short = _git("rev-parse", "--short", "feature/safe-ratio")
    cl_p = repo / "CHANGELOG.md"
    checks["changelog_updated"] = cl_p.is_file() and "## 1.2.0" in cl_p.read_text(encoding="utf-8") and bool(feat_short) and feat_short in cl_p.read_text(encoding="utf-8")
    # Functional check of cherry-picked fix in calc.py
    calc_ok = False
    calc_p = repo / "calc.py"
    if calc_p.is_file():
        res = subprocess.run(
            [sys.executable, "-c", "import calc; assert calc.compute_ratio(10, 0) == 0.0; assert calc.compute_ratio(1, 3) == 0.3333"],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )
        calc_ok = res.returncode == 0
    checks["cherry_pick_fix_verified"] = calc_ok
    return checks


def _eval_05_config_and_archive_ops(ws: Path) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}
    ini_p = ws / "deploy" / "app.prod.ini"
    ini_ok = False
    if ini_p.is_file():
        cfg = configparser.ConfigParser()
        cfg.read(ini_p, encoding="utf-8")
        ini_ok = (
            cfg.get("server", "host", fallback="") == "127.0.0.1"
            and cfg.get("server", "port", fallback="") == "8443"
            and cfg.get("server", "workers", fallback="") == "8"
            and cfg.get("server", "tls", fallback="") == "true"
            and cfg.get("logging", "level", fallback="") == "info"
            and cfg.get("logging", "format", fallback="") == "text"
        )
    checks["prod_ini_merged"] = ini_ok

    red_p = ws / "deploy" / "redacted.env"
    red_ok = False
    if red_p.is_file():
        txt = red_p.read_text(encoding="utf-8")
        red_ok = (
            "DB_PASSWORD=placeholder" not in txt
            and "API_TOKEN=placeholder" not in txt
            and "DB_PASSWORD=[REDACTED]" in txt
            and "API_TOKEN=[REDACTED]" in txt
            and "DB_HOST=db.internal" in txt
        )
    checks["secrets_redacted"] = red_ok

    bundle_p = ws / "dist" / "release-bundle.tar.gz"
    sha_p = ws / "dist" / "release-bundle.sha256"
    tar_ok = False
    if bundle_p.is_file():
        try:
            with tarfile.open(bundle_p, "r:gz") as tf:
                names = set(tf.getnames())
                tar_ok = {"app.prod.ini", "redacted.env", "assets/banner.txt", "assets/routes.json"}.issubset(names)
        except tarfile.TarError:
            tar_ok = False
    checks["tarball_valid"] = tar_ok
    checks["sha256_matches"] = (
        bundle_p.is_file()
        and sha_p.is_file()
        and sha_p.read_text(encoding="utf-8").strip().startswith(_sha256_file(bundle_p))
    )
    return checks


def _eval_06_document_markdown_report(ws: Path) -> Dict[str, bool]:
    md_p = ws / "reports" / "weekly_exec.md"
    json_p = ws / "reports" / "action_items.json"
    checks: Dict[str, bool] = {}
    md_ok = False
    if md_p.is_file():
        txt = md_p.read_text(encoding="utf-8")
        md_ok = (
            "## Executive Summary" in txt
            and "## Open Action Items" in txt
            and "## Completed Items" in txt
            and "## KPI Table" in txt
            and "-78.00" in txt
            and "+20.00" in txt
            and "-3.00" in txt
        )
    checks["markdown_structure_and_deltas"] = md_ok

    json_ok = False
    if json_p.is_file():
        try:
            data = json.loads(json_p.read_text(encoding="utf-8"))
            open_items = data.get("open", [])
            done_items = data.get("completed", [])
            json_ok = (
                len(open_items) == 3
                and len(done_items) == 2
                and [x["owner"] for x in open_items] == ["@alice", "@alice", "@carol"]
                and open_items[0]["due"] == "2026-11-27"
            )
        except (OSError, ValueError, KeyError, TypeError):
            json_ok = False
    checks["action_items_json_exact"] = json_ok
    return checks


def _eval_07_local_http_service_check(ws: Path) -> Dict[str, bool]:
    rep_p = ws / "web" / "probe_report.json"
    idx_sha = _sha256_file(ws / "web" / "static" / "index.html")
    api_sha = _sha256_file(ws / "web" / "static" / "api_data.json")
    checks: Dict[str, bool] = {}
    rep_ok = False
    lat_ok = False
    if rep_p.is_file():
        try:
            rep = json.loads(rep_p.read_text(encoding="utf-8"))
            routes = rep.get("routes", {})
            rep_ok = (
                isinstance(rep.get("port"), int)
                and rep["port"] > 0
                and rep.get("server_stopped") is True
                and routes.get("/index.html", {}).get("status") == 200
                and routes.get("/index.html", {}).get("sha256") == idx_sha
                and routes.get("/api_data.json", {}).get("status") == 200
                and routes.get("/api_data.json", {}).get("sha256") == api_sha
                and routes.get("/health", {}).get("status") == 200
            )
            lat_ok = all(
                isinstance(routes.get(r, {}).get("latency_ms"), (int, float))
                and routes[r]["latency_ms"] > 0.0
                for r in ("/index.html", "/api_data.json", "/health")
            )
        except (OSError, ValueError, KeyError, TypeError):
            rep_ok = False
    checks["http_probes_verified"] = rep_ok
    checks["http_latency_positive"] = lat_ok
    return checks


def _eval_08_permissions_and_cleanup_audit(ws: Path) -> Dict[str, bool]:
    key_p = ws / "sec" / "keys" / "service.key"
    script_p = ws / "sec" / "scripts" / "run.sh"
    out_p = ws / "sec" / "output.txt"
    sum_p = ws / "sec" / "audit_summary.json"
    checks: Dict[str, bool] = {}
    checks["key_mode_0600"] = key_p.is_file() and stat.S_IMODE(key_p.stat().st_mode) == 0o600
    checks["script_mode_0755"] = script_p.is_file() and stat.S_IMODE(script_p.stat().st_mode) == 0o755
    checks["script_executed"] = out_p.is_file() and out_p.read_text(encoding="utf-8").strip() == "hardened-run-ok"
    cache_dir = ws / "sec" / "cache"
    checks["cache_tmp_purged"] = (
        not list(cache_dir.glob("*.tmp")) and (cache_dir / "keep.dat").is_file()
    )
    sum_ok = False
    if sum_p.is_file():
        try:
            s = json.loads(sum_p.read_text(encoding="utf-8"))
            sum_ok = (
                s.get("key_mode") == "0o600"
                and s.get("script_mode") == "0o755"
                and s.get("removed_tmp_files") == ["stale_1.tmp", "stale_2.tmp"]
                and s.get("output_present") is True
            )
        except (OSError, ValueError, TypeError):
            sum_ok = False
    checks["audit_summary_exact"] = sum_ok
    return checks


def _eval_09_log_triage_incident_report(ws: Path) -> Dict[str, bool]:
    err_p = ws / "triage" / "error_lines.log"
    sum_p = ws / "triage" / "incident_summary.json"
    checks: Dict[str, bool] = {}
    err_ok = False
    if err_p.is_file():
        lines = [ln for ln in err_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        err_ok = (
            len(lines) == 5
            and lines[0].startswith("ts=2026-11-01T10:00:05Z")
            and lines[-1].startswith("ts=2026-11-03T12:01:00Z")
            and all(("level=ERROR" in ln or "level=CRITICAL" in ln) for ln in lines)
        )
    checks["error_log_chronological"] = err_ok
    sum_ok = False
    if sum_p.is_file():
        try:
            s = json.loads(sum_p.read_text(encoding="utf-8"))
            sum_ok = (
                s.get("total_error_lines") == 5
                and s.get("top_failing_service") == "payments"
                and s.get("service_5xx_counts") == {"auth": 1, "payments": 4}
                and s.get("code_5xx_counts") == {"500": 1, "502": 3, "503": 1}
                and s.get("retried_error_req_ids") == ["r2"]
            )
        except (OSError, ValueError, TypeError):
            sum_ok = False
    checks["incident_summary_exact"] = sum_ok
    return checks


def _eval_10_json_yaml_pipeline_patch(ws: Path) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}
    val_p = ws / "ci" / "validator.py"
    val_ok = False
    if val_p.is_file():
        res = subprocess.run(
            [sys.executable, str(val_p)],
            cwd=str(ws),
            capture_output=True,
            text=True,
        )
        val_ok = res.returncode == 0 and "validator-ok" in res.stdout
    checks["validator_passes"] = val_ok

    wf_p = ws / "ci" / "workflow.json"
    wf_ok = False
    if wf_p.is_file():
        try:
            wf = json.loads(wf_p.read_text(encoding="utf-8"))
            wf_ok = (
                wf["jobs"]["lint"]["timeout_minutes"] == 15
                and wf["jobs"]["test"]["timeout_minutes"] == 30
                and wf["jobs"]["package"]["timeout_minutes"] == 15
                and wf["jobs"]["lint"]["image"] == "hanzo/runner:v2-hardened"
            )
        except (OSError, ValueError, KeyError, TypeError):
            wf_ok = False
    checks["workflow_timeouts_and_tags"] = wf_ok

    diff_p = ws / "ci" / "migration_diff.json"
    diff_ok = False
    if diff_p.is_file():
        try:
            d = json.loads(diff_p.read_text(encoding="utf-8"))
            diff_ok = (
                "ci/workflow.json" in d
                and "ci/environments/prod.json" in d
                and "read_only_root" in d["ci/environments/prod.json"]
            )
        except (OSError, ValueError, TypeError):
            diff_ok = False
    checks["migration_diff_recorded"] = diff_ok
    return checks


# ---------------------------------------------------------------------------
# Canonical Manifest Fingerprint & External xlang-ai/OSWorld JSON Loader
# ---------------------------------------------------------------------------

def compute_manifest_digest(
    tasks: Sequence[OSWorldTaskSpec],
    external_tasks: Sequence[ExternalOSWorldTask] = (),
) -> str:
    """Compute canonical SHA-256 content digest across all task definitions."""
    payload = {
        "tasks": [
            {
                "task_id": t.task_id,
                "title": t.title,
                "category": t.category.value,
                "osworld_domain": t.osworld_domain,
                "instruction": t.instruction,
                "seed_files": t.seed_files,
                "readonly_assets": t.readonly_assets,
                "initial_modes": t.initial_modes,
                "setup_git_repo": t.setup_git_repo,
                "oracle_steps": [asdict(s) for s in t.oracle_steps],
                "evaluator_kind": t.evaluator_kind,
                "evaluator_spec": t.evaluator_spec,
            }
            for t in tasks
        ],
        "external_tasks": [asdict(et) for et in external_tasks],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_external_osworld_dir(
    external_dir: Path,
    *,
    tooling: Optional[DesktopToolingReport] = None,
) -> List[ExternalOSWorldTask]:
    """Load upstream xlang-ai/OSWorld Apache-2.0 JSON task specifications.

    Validates required OSWorld fields (``id``, ``instruction``, ``evaluator``)
    and classifies GUI-VM vs headless POSIX host compatibility transparently.
    """
    root = Path(external_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"External OSWorld task directory not found: {root}")

    loaded: List[ExternalOSWorldTask] = []
    json_files = sorted(root.rglob("*.json"))
    for fp in json_files:
        # Skip OSWorld's split index files like test_all.json / test_small.json
        if fp.name.startswith("test_") and fp.suffix == ".json":
            continue
        raw = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid OSWorld task JSON (expected object): {fp}")
        for req in ("id", "instruction", "evaluator"):
            if req not in raw:
                raise ValueError(f"OSWorld task JSON {fp} missing required key {req!r}")

        task_id = str(raw["id"])
        snapshot = str(raw.get("snapshot") or fp.parent.name or "os")
        instruction = str(raw["instruction"])
        related_apps = [str(a) for a in raw.get("related_apps", [snapshot])]
        evaluator = raw["evaluator"]
        if isinstance(evaluator, dict):
            eval_func = str(evaluator.get("func", "custom"))
        else:
            eval_func = str(evaluator)
        config_steps = len(raw.get("config", [])) if isinstance(raw.get("config"), list) else 0

        domain_tokens = {snapshot.lower(), *(a.lower() for a in related_apps)}
        requires_gui = bool(domain_tokens & GUI_ONLY_OSWORLD_DOMAINS)
        if not requires_gui:
            compatible = True
            reason = "headless_posix_compatible"
        elif tooling is not None and tooling.gui_stack_ready:
            compatible = True
            reason = "host_gui_stack_ready"
        else:
            compatible = False
            reason = "requires_desktop_vm_or_xvfb_gui_apps"

        loaded.append(
            ExternalOSWorldTask(
                task_id=task_id,
                source_path=str(fp.relative_to(root).as_posix()),
                snapshot=snapshot,
                instruction=instruction,
                related_apps=related_apps,
                evaluator_func=eval_func,
                requires_gui=requires_gui,
                compatible_with_host=compatible,
                compatibility_reason=reason,
                raw_config_steps=config_steps,
            )
        )
    return loaded
