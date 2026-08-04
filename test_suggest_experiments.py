"""Checks for /api/suggest-experiments.

Run against the local code:      python test_suggest_experiments.py
Run against the deployed engine: python test_suggest_experiments.py https://experiment-backend-s71q.onrender.com

Every well the engine returns has to be one a person can actually pipette:
inside each component's min/max, on that component's own min + n·step lattice
(unless a link derives it), and consistent with every component link.
"""
import sys
import types

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else None

if BASE:
    import json
    import urllib.request

    def post(payload):
        req = urllib.request.Request(BASE + "/api/suggest-experiments",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")
else:
    # lida_kinetics needs xgboost/libomp, which this endpoint does not — stub it out.
    from flask import Blueprint
    stub = types.ModuleType("lida_kinetics")
    stub.lida_bp = Blueprint("lida_stub", __name__)
    sys.modules.setdefault("lida_kinetics", stub)
    from app import app
    client = app.test_client()

    def post(payload):
        r = client.post("/api/suggest-experiments", json=payload)
        return r.status_code, r.get_json()

FAILURES = []

def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'' if ok else '  -> ' + detail}")
    if not ok:
        FAILURES.append(f"{label}: {detail}")

def cfg(**over):
    c = dict(anionName="A", anionMin=0, anionMax=6, anionStep=0.5, stockAnion=100, anionUnit="mM",
             cationName="B", cationMin=0, cationMax=6, cationStep=0.5, stockCation=100, cationUnit="mM",
             saltName="C", saltMin=0, saltMax=200, saltStep=10, stockSalt=1000, saltUnit="mM",
             enableCompD=False, compDMin=0, compDMax=1, compDStep=0.1, stockCompD=100,
             targetVolume=100, strategy="safe", numSuggestions=16, minDistanceFactor=0.05,
             dependencies=[], constants=[])
    c.update(over)
    return c

PTS = [(1, 0.5, 20, 0, 1), (2, 1, 40, 0.2, 1), (3, 2, 60, 0.4, 2), (4, 3, 80, 0.6, 2),
       (5, 4, 100, 0.8, 3), (6, 5, 120, 1.0, 3), (1.5, 5.5, 150, 0.5, 4), (2.5, 6, 200, 0.1, 4)]

def experiments():
    return [dict(sampleId=9000 + i, anion=a, cation=b, salt=c, compD=d, phase=p)
            for i, (a, b, c, d, p) in enumerate(PTS, 1)]

def suggest(c, n=16):
    return post({"config": c, "experiments": experiments(), "n_suggestions": n, "start_id": 9500})

def keys_of(c):
    return ["anion", "cation", "salt"] + (["compD"] if c["enableCompD"] else [])

def linked_targets(c):
    return {d["target"] for d in c["dependencies"]}

def on_grid(v, lo, step):
    return abs((v - lo) / step - round((v - lo) / step)) < 1e-6

def audit_bounds_and_grid(label, c, sugs):
    for k in keys_of(c):
        lo, hi, st = c[k + "Min"], c[k + "Max"], c[k + "Step"]
        vals = [s[k] for s in sugs]
        oob = [v for v in vals if v < lo - 1e-9 or v > hi + 1e-9]
        check(f"{label}: {k} inside [{lo}, {hi}]", not oob, f"{len(oob)} out of bounds e.g. {oob[:3]}")
        if k not in linked_targets(c):  # a derived value follows its source, not its own step
            off = [v for v in vals if not on_grid(v, lo, st)]
            check(f"{label}: {k} on the {st} lattice", not off, f"{len(off)} off-grid e.g. {off[:3]}")

def audit_links(label, c, sugs):
    for dep in c["dependencies"]:
        f, o = dep.get("factor", 1), dep.get("offset", 0)
        src, tgt = dep["source"], dep["target"]
        if dep.get("mode") == "range":
            fx, ox = dep.get("factorMax", f), dep.get("offsetMax", o)
            bad = [s for s in sugs
                   if not (min(s[src] * f + o, s[src] * fx + ox) - 1e-9 <= s[tgt]
                           <= max(s[src] * f + o, s[src] * fx + ox) + 1e-9)]
        elif dep.get("snapStep"):
            lo, st = c[tgt + "Min"], c[tgt + "Step"]
            bad = [s for s in sugs if abs(lo + round((s[src] * f + o - lo) / st) * st - s[tgt]) > 1e-6]
        else:
            bad = [s for s in sugs if abs(s[src] * f + o - s[tgt]) > 1e-6]
        check(f"{label}: link {src}->{tgt} honoured", not bad, f"{len(bad)}/{len(sugs)} violate it e.g. {bad[:2]}")

def audit_unique(label, c, sugs):
    wells = {tuple(s[k] for k in keys_of(c)) for s in sugs}
    check(f"{label}: no duplicate wells", len(wells) == len(sugs),
          f"{len(sugs)} returned but only {len(wells)} distinct")

def run(label, c, n=16, expect_status=200):
    code, data = suggest(c, n)
    check(f"{label}: HTTP {expect_status}", code == expect_status, f"got {code} {data.get('error', '')}")
    if code != 200:
        return data
    sugs = data["suggestions"]
    audit_bounds_and_grid(label, c, sugs)
    audit_links(label, c, sugs)
    audit_unique(label, c, sugs)
    return data


print("\n[1] plain 3D sweep, no links")
run("plain", cfg())

print("\n[2] fixed ratio link C = 5.3 x B  (the ratio must survive, exactly)")
run("ratio 5.3", cfg(dependencies=[dict(source="cation", target="salt", mode="fixed", factor=5.3, offset=0)]))

print("\n[3] same link, snapped onto C's own step grid")
run("ratio 5.3 snapped", cfg(dependencies=[dict(source="cation", target="salt", mode="fixed",
                                                factor=5.3, offset=0, snapStep=True)]))

print("\n[4] link that would push C past its max -> those wells must be dropped, not clipped")
run("ratio 50", cfg(dependencies=[dict(source="cation", target="salt", mode="fixed", factor=50, offset=0)]))

print("\n[5] range link, 4 components")
run("range link", cfg(enableCompD=True,
                      dependencies=[dict(source="salt", target="compD", mode="range",
                                         factor=0.002, offset=0, factorMax=0.006, offsetMax=0)]))

print("\n[6] chained links A -> B -> C")
run("chained", cfg(dependencies=[dict(source="anion", target="cation", mode="fixed", factor=0.5, offset=0),
                                 dict(source="cation", target="salt", mode="fixed", factor=20, offset=0)]))

print("\n[7] unsatisfiable link -> a clear 400, not a crash")
data = run("impossible", cfg(cationMin=0.5,
                             dependencies=[dict(source="cation", target="salt", mode="fixed",
                                                factor=1000, offset=0)]),
           expect_status=400)
check("impossible: explains itself", bool(data.get("error")), "no error message returned")

print("\n[8] fine step must not be silently coarsened")
data = run("fine step", cfg(anionStep=0.05, cationStep=1, saltStep=10), n=12)

print("\n[9] oversized grid: thinned on the user's own lattice, and it says so")
data = run("huge grid", cfg(anionStep=0.05, cationStep=0.05, saltStep=1), n=12)
check("huge grid: warns about thinning", any(w["axis"] in ("anion", "cation", "salt")
                                             for w in data.get("warnings", [])),
      f"warnings={data.get('warnings')}")

print("\n[10] risky strategy respects links too")
run("risky", cfg(strategy="risky",
                 dependencies=[dict(source="cation", target="salt", mode="fixed", factor=5.3, offset=0)]))

print("\n[11] collapsed axis (min == max)")
run("collapsed", cfg(saltMin=50, saltMax=50, saltStep=10))

print("\n" + ("FAILED: " + "; ".join(FAILURES) if FAILURES else "All checks passed."))
sys.exit(1 if FAILURES else 0)
