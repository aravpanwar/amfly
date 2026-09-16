"""Reconstruct the operator's drive for a run recorded before it was saved.

Compulsion.update() is a pure function of the chamber levels, and those ARE
recorded, so replaying it reproduces exactly what the operator received during
the run. This is a recomputation, not an estimate.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from amfly.wiring.compulsion import Compulsion

run = Path(sys.argv[1])
z = dict(np.load(run / "run.npz"))
heat = z["heat"][:, :5]
amp = max(float(heat.max()), 1e-9)
levels = heat / amp

comp = Compulsion(np.arange(4, dtype=np.int64), np.arange(4, dtype=np.int64), 8, 6)
drive = np.zeros((len(levels), 2), dtype=np.float32)
for t, lv in enumerate(levels):
    comp.update(lv)
    drive[t] = comp.state

z["drive"] = drive
np.savez_compressed(run / "run.npz", **z)
print("reward  min %.3f  max %.3f  mean %.3f" % (drive[:,0].min(), drive[:,0].max(), drive[:,0].mean()))
print("punish  min %.3f  max %.3f  mean %.3f" % (drive[:,1].min(), drive[:,1].max(), drive[:,1].mean()))
print("wrote drive into", run / "run.npz")
