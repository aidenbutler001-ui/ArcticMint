"""Fake nvidia-smi for the preflight test: prints $FAKE_SMI (default '0, 300\n1, 900') for --query-gpu=index,memory.used."""
import os, sys
print(os.environ.get("FAKE_SMI", "0, 300\n1, 900").replace("\\n", "\n"))
