import sys
import os

print("Python version:", sys.version)
print("\nsys.path:")
for path in sys.path:
    print("  -", path)

# Check awkward
try:
    import awkward as ak
    print("\nawkward version:", ak.__version__)
    print("awkward location:", ak.__file__)
except Exception as e:
    print("\nError importing awkward:", e)

# Check uproot
try:
    import uproot
    print("\nuproot version:", uproot.__version__)
    print("uproot location:", uproot.__file__)
except Exception as e:
    print("\nError importing uproot:", e)

# Check pandas
try:
    import pandas as pd
    print("\npandas version:", pd.__version__)
    print("pandas location:", pd.__file__)
except Exception as e:
    print("\nError importing pandas:", e)

# Check ROOT (PyROOT)
try:
    import ROOT
    # For ROOT, we use the gROOT object to retrieve the version.
    print("\nROOT version (gROOT):", ROOT.gROOT.GetVersion())
    # ROOT might not have a __file__ attribute if it's built into the system.
    if hasattr(ROOT, '__file__'):
        print("ROOT location:", ROOT.__file__)
    else:
        print("ROOT location: (not available via __file__ attribute)")
except Exception as e:
    print("\nError importing ROOT:", e)
