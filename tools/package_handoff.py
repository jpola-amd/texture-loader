#!/usr/bin/env python3
"""Small stdlib-only loader package manifest/preflight utility (Python 3.9+)."""
import argparse
import csv
import ctypes
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


class PackageError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def architecture(value):
    """One declared base ISA target; this is configuration, not qualification."""
    if value is None or value == "":
        raise PackageError("ARCHITECTURE_MISSING: declare one HIP_ARCHITECTURES target")
    if isinstance(value, (list, tuple)) or (isinstance(value, str) and (";" in value or "," in value)):
        raise PackageError("ARCHITECTURE_AMBIGUOUS: one base ISA target is required")
    if not isinstance(value, str) or not re.fullmatch(r"gfx[0-9a-f]{3,6}", value):
        raise PackageError(f"ARCHITECTURE_INVALID: expected one concrete base ISA name, got {value!r}")
    return value


def configured_architecture(metadata):
    configuration = metadata.get("configuration", {})
    if not isinstance(configuration, dict):
        raise PackageError("ARCHITECTURE_INVALID: configuration must be an object")
    return architecture(configuration.get("HIP_ARCHITECTURES"))


def bundle_targets(path):
    data = Path(path).read_bytes()
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    if not data.startswith(magic):
        raise PackageError(f"CODE_OBJECT_FORMAT: not a HIP/Clang bundle: {path}")
    offset = len(magic)
    count, = struct.unpack_from("<Q", data, offset)
    offset += 8
    if count > 64:
        raise PackageError("CODE_OBJECT_FORMAT: excessive bundle entries")
    targets = []
    for _ in range(count):
        start, size, length = struct.unpack_from("<QQQ", data, offset)
        offset += 24
        if start + size > len(data) or offset + length > len(data):
            raise PackageError("CODE_OBJECT_FORMAT: truncated bundle")
        target = data[offset:offset + length].decode("ascii").rstrip("\0")
        offset += length
        if target.startswith("hip"):
            targets.append(target)
    if not targets:
        raise PackageError("CODE_OBJECT_FORMAT: no HIP device image")
    return targets


def module_check(path, target):
    target = architecture(target)
    if not Path(path).is_file():
        raise PackageError(f"MODULE_MISSING: {path}")
    images = bundle_targets(path)
    if not any(t.split("--")[-1].split(":")[0] == target for t in images):
        raise PackageError(f"MODULE_TARGET_MISMATCH: expected {target}, found {images}")
    return images


def exports(path):
    """PE export names on Windows; native nm for ELF on Linux."""
    path = Path(path)
    if not path.is_file():
        raise PackageError(f"LIBRARY_MISSING: {path}")
    data = path.read_bytes()
    if data[:2] != b"MZ":
        result = subprocess.run(["nm", "-D", "--defined-only", str(path)], capture_output=True, text=True)
        if result.returncode:
            raise PackageError("EXPORT_INVENTORY: nm could not inspect library")
        return sorted({line.split()[-1] for line in result.stdout.splitlines() if line.split()})
    pe, = struct.unpack_from("<I", data, 0x3c)
    if data[pe:pe + 4] != b"PE\0\0":
        raise PackageError("EXPORT_INVENTORY: bad PE header")
    sections, = struct.unpack_from("<H", data, pe + 6)
    optional_size, = struct.unpack_from("<H", data, pe + 20)
    optional = pe + 24
    kind, = struct.unpack_from("<H", data, optional)
    export_rva, = struct.unpack_from("<I", data, optional + (112 if kind == 0x20b else 96))
    if not export_rva:
        return []
    table = optional + optional_size
    ranges = []
    for i in range(sections):
        virtual_size, address, raw_size, raw = struct.unpack_from("<IIII", data, table + 40 * i + 8)
        ranges.append((address, max(virtual_size, raw_size), raw))

    def position(rva):
        for address, size, raw in ranges:
            if address <= rva < address + size:
                return raw + rva - address
        raise PackageError(f"EXPORT_INVENTORY: invalid RVA {rva}")

    directory = position(export_rva)
    count, = struct.unpack_from("<I", data, directory + 24)
    names_rva, = struct.unpack_from("<I", data, directory + 32)
    names = position(names_rva)
    result = []
    for i in range(count):
        rva, = struct.unpack_from("<I", data, names + 4 * i)
        start = position(rva)
        result.append(data[start:data.index(0, start)].decode("ascii"))
    return sorted(result)


def probe(library, required):
    present = set(exports(library))
    missing = set(required) - present
    if missing:
        raise PackageError("REQUIRED_EXPORT_MISSING: " + ", ".join(sorted(missing)))
    directory = None
    try:
        if os.name == "nt":
            directory = os.add_dll_directory(str(Path(library).resolve().parent))
            dll = ctypes.WinDLL(str(Path(library).resolve()))
        else:
            dll = ctypes.CDLL(str(Path(library).resolve()))
        query = dll.hipDemandGetContractAbiV1
        query.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        query.restype = ctypes.c_uint32
        data = (ctypes.c_uint32 * 10)()
        if query(1, 40, data) != 0 or list(data) != [1, 40, 16, 32, 56, 32, 72, 8, 1, 0]:
            raise PackageError("ABI_MISMATCH: safe C layout query disagrees")
        sentinel = (ctypes.c_uint32 * 10)(*[0x12345678] * 10)
        for version, size in [(99, 40), (1, 39)]:
            if query(version, size, sentinel) != 15 or list(sentinel) != [0x12345678] * 10:
                raise PackageError("ABI_MISMATCH: invalid query not rejected before output access")
        return {"required_exports": len(required), "safe_query": list(data), "mismatch_queries_rejected": 2}
    finally:
        if directory is not None:
            directory.close()


def generate_fixtures(l01, l02, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cubic, refresh = load(l01), load(l02)
    dump(out / "cubic.json", cubic)
    for fixture in refresh["fixtures"]:
        original = Path(l02).parent / fixture["retained"]
        if sha(original) != fixture["sha256"]:
            raise PackageError("FIXTURE_HASH_MISMATCH")
        name = "files/" + original.name
        (out / "files").mkdir(exist_ok=True)
        shutil.copyfile(original, out / name)
        fixture["retained"] = name
        if "authored_mips" in fixture:
            levels = fixture["authored_mips"]
            floating = fixture["format"] == "FLOAT"
            while levels[-1]["width"] > 1 or levels[-1]["height"] > 1:
                previous = levels[-1]
                w, h = max(1, previous["width"] // 2), max(1, previous["height"] // 2)
                pixels = []
                for y in range(h):
                    for x in range(w):
                        selected = [previous["expected_point_rgba"][yy * previous["width"] + xx]
                                    for yy in range(2 * y, min(2 * y + 2, previous["height"]))
                                    for xx in range(2 * x, min(2 * x + 2, previous["width"]))]
                        if floating:
                            pixels.append([sum(p[c] for p in selected) / len(selected) for c in range(4)])
                        else:
                            pixels.append([(sum(round(p[c] * 255) for p in selected) + len(selected) // 2)
                                           // len(selected) / 255 for c in range(4)])
                levels.append({"original_level": len(levels), "width": w, "height": h,
                               "expected_point_rgba": pixels, "generated": True})
    refresh["sampling"] = {"spatial": "Point", "mip": "Point", "normalized_coordinates": True,
                           "uv": "each level's texel centers (x+.5)/Wm,(y+.5)/Hm",
                           "lod": "explicit original level m", "output_mask": 1, "derivatives": "not requested",
                           "complete": "Success/Complete, required original range [m,m]",
                           "cold": "Pending/Missing, zero fallback, process then complete retry",
                           "failed_refresh": "no publication; SourceFailure or DeviceOutOfMemory; new valid green owner recovers",
                           "lifetime": "finish, drain, discard context, destroy owner/source, overwrite same filename, reopen"}
    dump(out / "refresh.json", refresh)
    for source, record in [("impulse", cubic["impulse"]), ("palette", cubic["smart_point_transition"])]:
        levels = record["authored_mips"]
        data = bytearray(b"HDTREF1\0" + struct.pack("<4I", 8, 8, 1, len(levels)))
        for level in levels:
            for pixel in level["rgba_pixels"]:
                data.extend(struct.pack("<4f", *pixel))
        (out / f"consumer-{source}.hdtref").write_bytes(data)
    v, ds, dt = float(Fraction(5405, 18432)), float(Fraction(-299, 192)), float(Fraction(-1175, 384))
    definitions = [
        ("impulse", "bicubic", 0, 7, .4375, .4375, 0, 0, 4/9, 0, 0, 0, 0),
        ("impulse", "bicubic", 0, 7, .46875, .5, 0, 0, v, ds, dt, 0, 0),
        ("impulse", "smart-bicubic", 1, 7, .46875, .5, 0, 0, v, ds, dt, 0, 0),
        ("palette", "bicubic", 0, 7, .4375, .4375, 1, 0, 1, 0, 0, 1, 1),
        ("palette", "smart-bicubic", 1, 1, .4375, .4375, 0, .1875, .5, 0, 0, 0, 3),
        ("palette", "smart-bicubic", 1, 7, .4375, .4375, 0, .1875, 1, 0, 0, 0, 3)]
    with (out / "consumer-cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "mode", "gradient", "mask", "s", "t", "lod", "dxs", "dxt", "dys", "dyt",
                         "A", "jitterS", "jitterT", "value", "ds", "dt", "first", "last",
                         "valueBound", "normalizedDsBound", "normalizedDtBound"])
        for source, mode, grad, mask, s, t, lod, dx, value, ds, dt, first, last in definitions:
            writer.writerow([source, mode, grad, mask, s, t, lod, dx, 0, 0, 0, 1, 0, 0, value, ds, dt, first, last,
                             2e-5 * (1 + abs(value)), 1e-4 * (1 + abs(ds / 8)), 1e-4 * (1 + abs(dt / 8))])


def create_manifest(prefix, metadata, abi):
    prefix = Path(prefix).resolve()
    metadata = load(metadata)
    target = configured_architecture(metadata)
    library = "bin/hip_demand_texture.dll" if os.name == "nt" else "lib/libhip_demand_texture.so"
    names = exports(prefix / library)
    def public_export(name):
        if name == "hipDemandGetContractAbiV1":
            return True
        if os.name == "nt":
            return bool(re.match(r"^\?[^@]+@(DemandTextureLoader|Ticket)@hip_demand@@", name) or
                        re.match(r"^\?[^@]+@Texture@whole_mip_v1@hip_demand@@", name) or
                        re.match(r"^\?(getErrorString|setLogLevel|getLogLevel)@hip_demand@@", name) or
                        re.match(r"^\?allocateLoaderIncarnation@contract_v1@hip_demand@@", name))
        return "Impl" not in name and bool(re.match(
            r"^_ZNK?10hip_demand(19DemandTextureLoader|6Ticket|12whole_mip_v17Texture)", name))
    required = [n for n in names if public_export(n)]
    if "hipDemandGetContractAbiV1" not in required:
        raise PackageError("REQUIRED_EXPORT_MISSING: hipDemandGetContractAbiV1")
    inventory = {"declaration_headers": ["include/DemandLoading/DemandTextureLoader.h",
                                        "include/DemandLoading/WholeMipTexture.h", "include/DemandLoading/Contracts.h"],
                 "required": required, "all_actual_exports": names,
                 "signatures": "MSVC decorated symbols encode exact signatures; see matching installed declarations",
                 "legacy_calls_require_preflight": True,
                 "optional_OIIO_factory_present": any("createImageSource" in n for n in names)}
    dump(prefix / "share/hip-demand-texture/export-inventory.json", inventory)
    layouts = load(abi)
    dump(prefix / "share/hip-demand-texture/abi-inventory.json", layouts)
    modules = {}
    for path in prefix.rglob("*.co"):
        modules[path.relative_to(prefix).as_posix()] = {"targets": module_check(path, target)}
    entries = {"texture_sampling_kernel.co": ["sampleLegacyTextures"],
               "whole_mip_kernel.co": ["sampleWholeMipTextures"], "shared_contract_kernel.co": ["evaluateContract"],
               "cubic_kernel.co": ["sampleCubic", "sampleCubicNativePrimitive"],
               "handoff_consumer_kernel.co": ["installedCubicConsumer"]}
    for name, module in modules.items():
        module["entry_points"] = entries.get(Path(name).name, [])
    manifest = {"schema": "hip-demand-handoff-manifest-v1", "metadata": metadata,
                "host_library": library, "abi": layouts, "required_exports": required,
                "device_interface": {"header_only": True, "include_order": ["hip/hip_runtime.h", "DemandLoading/CubicSampling.h"],
                                     "flags": ["-O3", "--std=c++17", "-DHIP_ENABLE_WARP_SYNC_BUILTINS"],
                                     "architecture": target, "modules": modules},
                "files": {p.relative_to(prefix).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                          for p in sorted(prefix.rglob("*")) if p.is_file() and p != prefix / "manifest.json"}}
    dump(prefix / "manifest.json", manifest)
    return {"files": len(manifest["files"]), "required_exports": len(required), "code_objects": len(modules)}


def verify(prefix):
    prefix = Path(prefix).resolve()
    manifest = load(prefix / "manifest.json")
    if manifest.get("schema") != "hip-demand-handoff-manifest-v1":
        raise PackageError("MANIFEST_SCHEMA_MISMATCH")
    target = architecture(manifest["device_interface"].get("architecture"))
    if target != configured_architecture(manifest["metadata"]):
        raise PackageError("ARCHITECTURE_MISMATCH: device interface differs from configured target")
    actual = {p.relative_to(prefix).as_posix() for p in prefix.rglob("*") if p.is_file() and p != prefix / "manifest.json"}
    unexpected = actual - set(manifest["files"])
    if unexpected:
        raise PackageError("UNLISTED_ARTIFACT: " + ", ".join(sorted(unexpected)))
    for relative, expected in manifest["files"].items():
        path = (prefix / relative).resolve()
        if not path.is_relative_to(prefix):
            raise PackageError("MANIFEST_PATH_ESCAPE")
        if not path.is_file():
            raise PackageError(f"ARTIFACT_MISSING: {relative}")
        if sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise PackageError(f"MANIFEST_HASH_MISMATCH: {relative}")
    for relative, expected in manifest["metadata"].get("source_header_hashes", {}).items():
        if sha(prefix / relative) != expected:
            raise PackageError(f"HEADER_SOURCE_MISMATCH: {relative}")
    for relative in manifest["device_interface"]["modules"]:
        module_check(prefix / relative, target)
    abi = probe(prefix / manifest["host_library"], manifest["required_exports"])
    return {"result": "consistent", "files": len(manifest["files"]), "abi": abi, "architecture": target,
            "qualification": manifest["metadata"].get("qualification", "not specified")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fixtures"); f.add_argument("--l01", required=True); f.add_argument("--l02", required=True); f.add_argument("--out", required=True)
    m = sub.add_parser("manifest"); m.add_argument("prefix"); m.add_argument("--metadata", required=True); m.add_argument("--abi", required=True)
    v = sub.add_parser("verify"); v.add_argument("prefix")
    p = sub.add_parser("probe"); p.add_argument("library"); p.add_argument("--require", action="append", default=[])
    c = sub.add_parser("module"); c.add_argument("path"); c.add_argument("--target", default="gfx1201")
    r = sub.add_parser("run"); r.add_argument("prefix"); r.add_argument("executable"); r.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "fixtures":
        generate_fixtures(args.l01, args.l02, args.out); result = {"result": "fixtures serialized"}
    elif args.command == "manifest":
        result = create_manifest(args.prefix, args.metadata, args.abi)
    elif args.command == "verify":
        result = verify(args.prefix)
    elif args.command == "probe":
        result = probe(args.library, ["hipDemandGetContractAbiV1"] + args.require)
    elif args.command == "module":
        result = {"targets": module_check(args.path, args.target)}
    else:
        checked = verify(args.prefix)
        prefix = Path(args.prefix).resolve()
        name = args.executable + (".exe" if os.name == "nt" and not args.executable.endswith(".exe") else "")
        executable = prefix / "bin" / name
        if executable.parent != prefix / "bin" or not executable.is_file():
            raise PackageError("EXECUTABLE_MISSING_OR_OUTSIDE_PACKAGE")
        if "--module" in args.args:
            index = args.args.index("--module")
            module_check(args.args[index + 1], checked["architecture"])
        env = os.environ.copy()
        env["HIP_DEMAND_TEST_KERNEL_DIR"] = str(prefix / "bin")
        env["HDT_TEST_FILE_ROOT"] = str(prefix / "test-work")
        env["HDT_TEST_LOADER_LIBRARY"] = str(prefix / ("bin/hip_demand_texture.dll" if os.name == "nt" else "lib/libhip_demand_texture.so"))
        env["HDT_TEST_MODULE_IDENTITY"] = "1"
        (prefix / "test-work").mkdir(exist_ok=True)
        return subprocess.call([str(executable), *args.args], cwd=prefix / "test-work", env=env)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (PackageError, OSError, ValueError, struct.error, IndexError, KeyError) as error:
        print("PACKAGE_ERROR: " + str(error), file=sys.stderr)
        sys.exit(2)
