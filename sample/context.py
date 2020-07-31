import os
import sys

if os.path.abspath('../model') not in sys.path:
    sys.path.insert(0, os.path.abspath('../model'))