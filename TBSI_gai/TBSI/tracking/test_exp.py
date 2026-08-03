"""Compatibility wrapper for the safe tracker evaluation entrypoint.

Historically this file hard-coded unrelated datasets and bypassed the
result-resume guard. Keep the filename for old scripts, but route all behavior
through tracking/test.py.
"""

from test import main


if __name__ == '__main__':
    main()
