#!/usr/bin/env python3
"""
inspect_onnx_io.py — RAM-safe ONNX graph I/O inspector.

Reads an ONNX ModelProto by streaming the protobuf wire format directly, so it
can extract graph inputs/outputs, opset, and an op_type histogram from models
that are far larger than available memory (the SF3D backbone is ~900 MB). Large
sub-messages (node bodies, initializer blobs) are seeked over, never held.

Usage:
    python3 tools/inspect_onnx_io.py model.onnx [model2.onnx ...]
    python3 tools/inspect_onnx_io.py --json out.json model.onnx ...

Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# ONNX TensorProto.DataType enum
ELEM_TYPE = {
    0: "UNDEFINED", 1: "FLOAT", 2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16",
    6: "INT32", 7: "INT64", 8: "STRING", 9: "BOOL", 10: "FLOAT16", 11: "DOUBLE",
    12: "UINT32", 13: "UINT64", 14: "COMPLEX64", 15: "COMPLEX128", 16: "BFLOAT16",
}


# ---------- in-memory protobuf field iterator (for small captured messages) ----------
def _read_varint_buf(data, i):
    shift = 0
    result = 0
    while True:
        b = data[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, i


def fields(data):
    """Yield (field_number, wire_type, value) for a small in-memory protobuf message."""
    i = 0
    n = len(data)
    while i < n:
        tag, i = _read_varint_buf(data, i)
        fn = tag >> 3
        wt = tag & 7
        if wt == 0:
            v, i = _read_varint_buf(data, i)
            yield fn, wt, v
        elif wt == 2:
            ln, i = _read_varint_buf(data, i)
            yield fn, wt, data[i:i + ln]
            i += ln
        elif wt == 5:
            yield fn, wt, data[i:i + 4]
            i += 4
        elif wt == 1:
            yield fn, wt, data[i:i + 8]
            i += 8
        else:
            raise ValueError(f"unsupported wire type {wt} at offset {i}")


def parse_valueinfo(data):
    """ValueInfoProto -> {name, elem_type, shape:[dim_value|dim_param,...]}"""
    name = None
    elem = None
    dims = []
    for fn, wt, v in fields(data):
        if fn == 1 and wt == 2:
            name = v.decode("utf-8", "replace")
        elif fn == 2 and wt == 2:  # TypeProto
            for fn2, wt2, v2 in fields(v):
                if fn2 == 1 and wt2 == 2:  # tensor_type
                    for fn3, wt3, v3 in fields(v2):
                        if fn3 == 1 and wt3 == 0:
                            elem = v3
                        elif fn3 == 2 and wt3 == 2:  # shape
                            for fn4, wt4, v4 in fields(v3):
                                if fn4 == 1 and wt4 == 2:  # Dimension
                                    dv = None
                                    dp = None
                                    for fn5, wt5, v5 in fields(v4):
                                        if fn5 == 1 and wt5 == 0:
                                            dv = v5
                                        elif fn5 == 2 and wt5 == 2:
                                            dp = v5.decode("utf-8", "replace")
                                    dims.append(dp if dp is not None else dv)
    return {
        "name": name,
        "elem_type": ELEM_TYPE.get(elem, elem),
        "shape": dims,
    }


def parse_opset(data):
    domain = ""
    version = None
    for fn, wt, v in fields(data):
        if fn == 1 and wt == 2:
            domain = v.decode("utf-8", "replace")
        elif fn == 2 and wt == 0:
            version = v
    return {"domain": domain or "ai.onnx", "version": version}


def parse_node_optype(data):
    for fn, wt, v in fields(data):
        if fn == 4 and wt == 2:  # op_type
            return v.decode("utf-8", "replace")
    return None


# ---------- streaming reader over a seekable file ----------
class Stream:
    def __init__(self, f):
        self.f = f

    def varint(self):
        shift = 0
        result = 0
        while True:
            b = self.f.read(1)
            if not b:
                return None
            bb = b[0]
            result |= (bb & 0x7F) << shift
            if not (bb & 0x80):
                return result
            shift += 7

    def take(self, n):
        return self.f.read(n)

    def skip(self, n):
        self.f.seek(n, os.SEEK_CUR)


def inspect(path):
    f = open(path, "rb")
    s = Stream(f)
    info = {
        "path": path,
        "bytes": os.path.getsize(path),
        "ir_version": None,
        "producer": None,
        "opsets": [],
        "inputs": [],
        "outputs": [],
        "node_count": 0,
        "initializer_count": 0,
        "op_types": {},
    }
    while True:
        tag = s.varint()
        if tag is None:
            break
        fn = tag >> 3
        wt = tag & 7
        if wt == 0:
            v = s.varint()
            if fn == 1:
                info["ir_version"] = v
            elif fn == 5:
                info["model_version"] = v
        elif wt == 2:
            ln = s.varint()
            if fn == 2:
                info["producer"] = s.take(ln).decode("utf-8", "replace")
            elif fn == 3:
                pv = s.take(ln).decode("utf-8", "replace")
                info["producer"] = (info["producer"] or "") + (" " + pv if pv else "")
            elif fn == 8:
                info["opsets"].append(parse_opset(s.take(ln)))
            elif fn == 7:  # graph
                start = f.tell()
                end = start + ln
                while f.tell() < end:
                    gtag = s.varint()
                    if gtag is None:
                        break
                    gfn = gtag >> 3
                    gwt = gtag & 7
                    if gwt == 0:
                        s.varint()
                    elif gwt == 2:
                        gln = s.varint()
                        if gfn == 11:
                            info["inputs"].append(parse_valueinfo(s.take(gln)))
                        elif gfn == 12:
                            info["outputs"].append(parse_valueinfo(s.take(gln)))
                        elif gfn == 1:  # node
                            info["node_count"] += 1
                            if gln <= 8192:
                                ot = parse_node_optype(s.take(gln))
                                if ot:
                                    info["op_types"][ot] = info["op_types"].get(ot, 0) + 1
                            else:
                                s.skip(gln)
                        elif gfn == 5:  # initializer
                            info["initializer_count"] += 1
                            s.skip(gln)
                        else:
                            s.skip(gln)
                    elif gwt == 5:
                        s.skip(4)
                    elif gwt == 1:
                        s.skip(8)
                    else:
                        raise ValueError(f"graph wire type {gwt}")
                f.seek(end, os.SEEK_SET)
            else:
                s.skip(ln)
        elif wt == 5:
            s.skip(4)
        elif wt == 1:
            s.skip(8)
        else:
            raise ValueError(f"top wire type {wt}")
    f.close()
    # tidy op_types: keep sorted by count desc
    info["op_types"] = dict(sorted(info["op_types"].items(), key=lambda kv: -kv[1]))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()
    results = []
    for m in args.models:
        try:
            r = inspect(m)
        except Exception as e:
            r = {"path": m, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        print(f"\n=== {os.path.basename(m)} ({r.get('bytes',0)/1e6:.1f} MB) ===")
        if "error" in r:
            print("  ERROR:", r["error"])
            continue
        print(f"  producer: {r.get('producer')} | ir_version: {r.get('ir_version')} | opsets: {r.get('opsets')}")
        print(f"  nodes: {r['node_count']} | initializers: {r['initializer_count']}")
        print("  INPUTS:")
        for x in r["inputs"]:
            print(f"     - {x['name']}: {x['elem_type']} {x['shape']}")
        print("  OUTPUTS:")
        for x in r["outputs"]:
            print(f"     - {x['name']}: {x['elem_type']} {x['shape']}")
        top = list(r["op_types"].items())[:12]
        print(f"  top op_types: {top}")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
