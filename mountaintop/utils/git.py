import os
import subprocess

# Return the git revision as a string
def git_version(short=False):
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        if len(out) < 1:
            # re-run command without env
            out = subprocess.Popen(cmd, stdout = subprocess.PIPE).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
        if short:
            GIT_REVISION = GIT_REVISION[:8]
    except OSError:
        GIT_REVISION ="unknown"

    return GIT_REVISION
