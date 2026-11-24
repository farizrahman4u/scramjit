"""The `scramjit` command line. Standard library only: argument parsing, help, suggestions, colors
and shell completion are in this package, built the same way as cimo's."""

import os
import sys

__all__ = ['main']


class Context:
    """What a command gets: the parsed values, the command, and styled stdout/stderr."""

    def __init__(self, command, values, out, err):
        self.command, self.values, self.out, self.err = command, values, out, err


def _color_mode(argv):
    mode = os.environ.get('SCRAMJIT_COLOR', 'auto')
    for i, a in enumerate(argv):
        if a == '--':
            break
        if a.startswith('--color='):
            mode = a.split('=', 1)[1]
        elif a == '--color' and i + 1 < len(argv):
            mode = argv[i + 1]
    return mode if mode in ('auto', 'always', 'never') else 'auto'


def _split_run(argv):
    """`--run` takes the rest of the command line: the program and its own arguments."""
    if '--run' not in argv or '--' in argv[:argv.index('--run')]:
        return argv, None
    i = argv.index('--run')
    return argv[:i], argv[i + 1:]


def main(argv=None):
    """Run the scramjit command line; returns the exit status."""
    from cimo.cli.term import Terminal
    from cimo.cli.parse import parse, render_help, usage_line, UsageError
    from . import commands
    argv = sys.argv[1:] if argv is None else list(argv)
    mode = _color_mode(argv)
    out, err = Terminal(sys.stdout, mode), Terminal(sys.stderr, mode)
    root = commands.build_tree()
    try:
        argv, program = _split_run(argv)
        p = parse(root, argv)
        cmd, values = p.command, p.values
        if values.get('version') and cmd is root:
            out.write(commands.version_text())
            return 0
        if values.get('help') or (cmd.func is None) or (cmd is root and not argv):
            out.write(render_help(cmd, out))
            return 0
        if program is not None:
            if cmd.name != 'preflight':
                raise UsageError('--run belongs to `preflight`', cmd)
            if not program:
                raise UsageError('--run needs a program', cmd, hint='`scramjit preflight --run app.py`, or `--run -m pytest`')
            values['run'] = program
        return cmd.func(Context(cmd, values, out, err)) or 0
    except UsageError as e:
        cmd = e.command or root
        err.write('%s %s' % (err.style('error:', 'error'), e.message))
        if e.suggestions:
            quoted = ' or '.join(err.style(s, 'bold') for s in e.suggestions)
            err.write('')
            err.write('  Did you mean %s?' % quoted)
        if e.hint:
            err.write('')
            err.write('  %s %s' % (err.style('hint:', 'hint'), err.code(e.hint)))
        err.write('')
        err.write(usage_line(cmd, err))
        err.write(err.code('Run `%s --help` for more.' % cmd.path))
        return 2
    except commands.Failure as f:
        return f.status
    except KeyboardInterrupt:
        err.write('')
        return 130
    except BrokenPipeError:
        try:
            sys.stdout = open(os.devnull, 'w')
        except OSError:
            pass
        return 141
