"""Map library+offset lines from a SEGV_BT dump to the nearest preceding symbol (nm, demangled)."""
import re, subprocess, sys, bisect
cache = {}
def syms(lib):
    if lib not in cache:
        out = []
        for flags in (['-DC', '--defined-only'], ['-C']):
            r = subprocess.run(['nm'] + flags + [lib], capture_output=True, text=True)
            for line in r.stdout.splitlines():
                p = line.split(' ', 2)
                if len(p) == 3 and p[1] in 'tTwW':
                    try: out.append((int(p[0], 16), p[2]))
                    except ValueError: pass
        out.sort(); cache[lib] = out
    return cache[lib]
for line in open(sys.argv[1], errors='replace'):
    m = re.match(r'#(\d+) \S+ (\S+\.so[^+]*)\+0x([0-9a-f]+)', line)
    if not m: continue
    lib, off = m.group(2), int(m.group(3), 16)
    s = syms(lib); i = bisect.bisect_right([a for a, _ in s], off) - 1
    name = '%s +0x%x' % (s[i][1][:200], off - s[i][0]) if i >= 0 else '?'
    print('#%s %s+0x%x  %s' % (m.group(1), lib.split('/')[-1], off, name))
