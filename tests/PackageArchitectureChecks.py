"""CPU-only controls for the package's declared target; no device execution."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prefix", type=Path, required=True)
parser.add_argument("--alternative-module", type=Path, required=True)
parser.add_argument("--work", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
assert args.work.is_absolute() and not args.work.exists(), "Use a fresh, test-owned project directory"
args.work.mkdir(parents=True)
tool = args.prefix / "share/hip-demand-texture/package_handoff.py"
abi = args.prefix / "share/hip-demand-texture/abi-inventory.json"
control = args.work / "control"
(control / "bin").mkdir(parents=True)
for source in (args.prefix / "bin").glob("*.dll"):
    shutil.copyfile(source, control / "bin" / source.name)
if os.name != "nt":
    shutil.copytree(args.prefix / "lib", control / "lib")
exe = "handoff_consumer.exe" if os.name == "nt" else "handoff_consumer"
shutil.copyfile(args.prefix / "bin" / exe, control / "bin" / exe)
shutil.copyfile(args.alternative_module, control / "bin" / "probe.co")
metadata = args.work / "metadata.json"
results = []
env = dict(os.environ, HIP_VISIBLE_DEVICES="-1")

def run(name, command, expected, diagnostic):
    command = [str(c) for c in command]
    if command[0] == "python":
        command[0] = sys.executable
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    output = result.stdout + result.stderr
    (args.work / (name + ".log")).write_text(output, encoding="utf8")
    assert result.returncode == expected and diagnostic in output, (name, result.returncode, output)
    results.append({"name": name, "command": command, "exit": result.returncode,
                    "expected_exit": expected, "diagnostic": diagnostic, "result": "passed"})

def meta(target):
    metadata.write_text(json.dumps({"configuration": {"HIP_ARCHITECTURES": target},
        "qualification": "CPU-only target selection control; no hardware qualification"}))

def manifest():
    return ["python", tool, "manifest", control, "--metadata", metadata, "--abi", abi]

try:
    for label, value, diagnostic in [
        ("missing", None, "ARCHITECTURE_MISSING"), ("empty", "", "ARCHITECTURE_MISSING"),
        ("multiple", "gfx1030;gfx1201", "ARCHITECTURE_AMBIGUOUS"),
        ("list", ["gfx1030", "gfx1201"], "ARCHITECTURE_AMBIGUOUS"),
        ("invalid", "not-a-target", "ARCHITECTURE_INVALID"),
        ("wrong-type", 1201, "ARCHITECTURE_INVALID")]:
        meta(value)
        run(label, manifest(), 2, diagnostic)
    meta("gfx1201")
    run("wrong-configured-target", manifest(), 2, "MODULE_TARGET_MISMATCH")
    meta("gfx1030")
    run("declared-alternative", manifest(), 0, "code_objects")
    data = json.loads((control / "manifest.json").read_text())
    assert data["device_interface"]["architecture"] == "gfx1030"
    run("verify-alternative", ["python", tool, "verify", control], 0, "gfx1030")
    # The actual supplied module matches the declared alternative. --abi never
    # initializes HIP; this proves run does not silently reimpose gfx1201.
    run("run-declared-alternative", ["python", tool, "run", control, "handoff_consumer",
        "--abi", "--module", args.alternative_module], 0, "pointer_bytes")
    run("run-wrong-for-declaration", ["python", tool, "run", control, "handoff_consumer",
        "--abi", "--module", args.prefix / "bin/handoff_consumer_kernel.co"], 2, "MODULE_TARGET_MISMATCH")
    data["device_interface"]["architecture"] = "gfx1201"
    (control / "manifest.json").write_text(json.dumps(data))
    run("conflicting-declaration", ["python", tool, "verify", control], 2, "ARCHITECTURE_MISMATCH")
finally:
    args.report.write_text(json.dumps({"cpu_only": True, "HIP_VISIBLE_DEVICES": "-1",
        "other_device_execution": False, "checks": results}, indent=2) + "\n")
    shutil.rmtree(control)
print(f"Passed {len(results)} CPU-only architecture controls")
