"""A scramjit function's compiled code moves between processes as cimo cache entries, as the LLM demo moves it
between its workers and between visits."""
import os
import subprocess
import sys

import pytest

KERNEL = '''
import numpy as np
from scramjit import scram

@scram
def axpy(a, x, y):
    for i in range(len(x)):
        y[i] += a * x[i]

x, y = np.arange(4.0), np.ones(4)
axpy(2.0, x, y)
assert list(y) == [1, 3, 5, 7]
module = next(v for v in axpy._cache.values() if hasattr(v, 'source')).module
'''

EXPORT = KERNEL + '''
open(sys.argv[1], 'wb').write(module.cache_entry())
print(module.cached)
'''

IMPORT = '''
import cimo
cimo.add_cache_entry(open(sys.argv[1], 'rb').read())
''' + KERNEL + '''
print(module.cached)
'''


def _run(tmp_path, name, code, cache):
    script = tmp_path / (name + '.py')
    script.write_text('import sys\n' + code)
    env = dict(os.environ, CIMO_CACHE_DIR=str(tmp_path / cache), PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    r = subprocess.run([sys.executable, str(script), str(tmp_path / 'entry')], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_entry_moves_compiled_code_to_a_fresh_cache(tmp_path):
    if sys.platform == 'emscripten':
        pytest.skip('needs subprocess')
    assert _run(tmp_path, 'export', EXPORT, 'first') == 'False'      # compiled
    assert _run(tmp_path, 'import', IMPORT, 'second') == 'True'      # another empty cache: loaded, nothing compiled
