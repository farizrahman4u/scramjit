"""Copy what the site's live demo loads into Pyodide: cimo as one file and scramjit's package.
Run after changing scramjit (or to pick up a newer cimo):  python3 tools/sync_site.py
cimo is the last commit of ../cimo next to this repository, never its uncommitted work, or the commit or tag in
CIMO_REF (a release: CIMO_REF=v0.0.5); without a checkout, whatever cimo the environment has.
"""

import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
OUT = os.path.join(ROOT, 'docs', 'py')


def committed_cimo(tmp):
    """cimo's last commit, exported: a checkout's uncommitted work must not reach the site. None if
    there is no checkout next to this repository."""
    repo = os.path.join(ROOT, '..', 'cimo')
    if not os.path.isdir(os.path.join(repo, '.git')):
        return None
    ref = os.environ.get('CIMO_REF', 'HEAD')
    archive = subprocess.run(['git', '-C', repo, 'archive', ref, 'python'], check=True, capture_output=True).stdout
    tarfile.open(fileobj=io.BytesIO(archive)).extractall(tmp)
    return tmp


def main():
    with tempfile.TemporaryDirectory() as tmp:
        sync(committed_cimo(tmp))


def sync(cimo_root):
    os.makedirs(os.path.join(OUT, 'scramjit'), exist_ok=True)
    for f in os.listdir(os.path.join(ROOT, 'scramjit')):
        if f.endswith('.py'):
            shutil.copy(os.path.join(ROOT, 'scramjit', f), os.path.join(OUT, 'scramjit', f))
    env = dict(os.environ)
    if cimo_root:
        env['PYTHONPATH'] = os.path.join(cimo_root, 'python')
    subprocess.run([sys.executable, '-m', 'cimo', 'onefile', '-o', os.path.join(OUT, 'cimo.py')], check=True, env=env)
    print('synced', sorted(os.listdir(os.path.join(OUT, 'scramjit'))), 'and cimo.py',
          os.path.getsize(os.path.join(OUT, 'cimo.py')) // 1024, 'KB')


if __name__ == '__main__':
    main()
