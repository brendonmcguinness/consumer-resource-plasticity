#!usrbinenv python3
"""Static checker for missing function references in utils.py and Community.py."""

import ast
import os
import sys
import glob

# Ensure current directory modules are importable
sys.path.insert(0, os.getcwd())

try:
    import utils
except ImportError:
    print("Error: Could not import 'utils' module. Make sure this script is run from the project root.")
    sys.exit(1)

try:
    import Community
except ImportError:
    print("Error: Could not import 'Community' module. Make sure this script is run from the project root.")
    sys.exit(1)


# Find every .py in the repo root, but skip the modules under test and the checker itself
all_py = glob.glob(os.path.join(os.getcwd(), "*.py"))
SCRIPTS = [
    os.path.basename(p) for p in all_py
    if os.path.basename(p) not in (
        "utils.py",
        "Community.py",
        os.path.basename(__file__)  # static_check.py
    )
]


missing = []

for script in SCRIPTS:
    path = os.path.join(os.getcwd(), script)
    if not os.path.isfile(path):
        print(f"Warning: '{script}' not found, skipping.")
        continue

    with open(path, 'r') as f:
        tree = ast.parse(f.read(), filename=script)

    for node in ast.walk(tree):
        # Look for calls like utils.foo() or Community.bar()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value = node.func.value
            if isinstance(value, ast.Name) and value.id in ("utils", "Community"):
                module = utils if value.id == "utils" else Community
                attr = node.func.attr
                if not hasattr(module, attr):
                    missing.append(f"{script}: {value.id}.{attr}() not found in module")

if missing:
    print("\nMissing references in utils.py  Community.py:")
    for msg in missing:
        print(f"  - {msg}")
    sys.exit(1)
else:
    print("No missing references detected in utils.py or Community.py.")
