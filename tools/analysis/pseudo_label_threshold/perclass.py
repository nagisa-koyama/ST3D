import pickle, numpy as np
d = pickle.load(open('%s/audit_full.pkl' % __import__('sys').argv[1], 'rb'))
C = ['Car', 'Pedestrian', 'Cyclist']; R = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70']
A = lambda recs: np.array(recs, dtype=float).reshape(-1, 4)
tg = {k: A(v) for k, v in d['target'].items()}; sr = {k: A(v) for k, v in d['source'].items()}
def m(a, ri, c):
    s = (a[:, 0] == ri) & (a[:, 2] >= 1) & ((a[:, 1] == c) if c else True)
    return a[s, 2].mean() if s.any() else np.nan, s.sum()
for s, a_s in sr.items():
    print('\n== %s: tgt/src mean pts per box, per class per ring (target GT | pseudo) ==' % s)
    print('ring   ' + ''.join('%-22s' % c for c in C) + 'ALL-classes pooled')
    for ri, rn in enumerate(R):
        row = '%-7s' % rn
        for c in [1, 2, 3, 0]:
            ms, ns = m(a_s, ri, c); mg, _ = m(tg['gt'], ri, c); mp, _ = m(tg['pseudo'], ri, c)
            row += '%5.3f | %5.3f (n=%4d)   ' % (mg / ms, mp / ms, ns) if ns >= 20 else '   --  (n=%4d)        ' % ns
        print(row)
print('\n== class mix of occupied boxes (share Car/Ped/Cyc) ==')
def mix(a):
    a = a[a[:, 2] >= 1]; return ' / '.join('%.2f' % ((a[:, 1] == c).mean()) for c in (1, 2, 3))
for k, a in list(sr.items()) + [('nuScenes GT', tg['gt']), ('nuScenes pseudo', tg['pseudo'])]:
    print('%-16s %s' % (k, mix(a)))
