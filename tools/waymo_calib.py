"""Minimal Waymo tfrecord reader: pulls Context.laser_calibrations only.
No tensorflow, no waymo_open_dataset. tfrecord = [uint64 len][crc][payload][crc]."""
import struct, numpy as np
from pathlib import Path

def _varint(b,i):
    v=0; s=0
    while True:
        x=b[i]; i+=1; v |= (x&0x7f)<<s
        if not x&0x80: return v,i
        s+=7
def _fields(b,start=0,end=None):
    i=start; end=len(b) if end is None else end
    while i<end:
        key,i=_varint(b,i); fn,wt=key>>3,key&7
        if wt==0:   v,i=_varint(b,i); yield fn,wt,v
        elif wt==1: yield fn,wt,b[i:i+8]; i+=8
        elif wt==2: ln,i=_varint(b,i); yield fn,wt,b[i:i+ln]; i+=ln
        elif wt==5: yield fn,wt,b[i:i+4]; i+=4
        else: raise ValueError(f'wire type {wt}')
def _doubles(fn_wanted, blob):
    """repeated double, packed (wt2) or unpacked (wt1)"""
    out=[]
    for fn,wt,v in _fields(blob):
        if fn!=fn_wanted: continue
        if wt==1: out.append(struct.unpack('<d',v)[0])
        elif wt==2: out += list(struct.unpack('<%dd'%(len(v)//8), v))
    return out
def first_frame(path):
    with open(path,'rb') as f:
        ln=struct.unpack('<Q',f.read(8))[0]; f.read(4)
        return f.read(ln)
NAMES={0:'UNKNOWN',1:'TOP',2:'FRONT',3:'SIDE_LEFT',4:'SIDE_RIGHT',5:'REAR'}
def laser_calibrations(path):
    frame=first_frame(path)
    ctx=next(v for fn,wt,v in _fields(frame) if fn==1 and wt==2)      # Frame.context = 1
    out={}
    for fn,wt,v in _fields(ctx):
        if fn!=3 or wt!=2: continue                                   # Context.laser_calibrations = 3
        name=None; inc=[]; lo=hi=None; ext=None
        for f2,w2,v2 in _fields(v):
            if   f2==1 and w2==0: name=NAMES.get(v2,str(v2))          # name
            elif f2==2:           inc += _doubles(2, bytes([0x11])+b'') if False else []
            elif f2==3 and w2==1: lo=struct.unpack('<d',v2)[0]        # beam_inclination_min
            elif f2==4 and w2==1: hi=struct.unpack('<d',v2)[0]        # beam_inclination_max
            elif f2==5 and w2==2: ext=_doubles(1, v2)                 # extrinsic -> Transform.transform = 1
        inc=_doubles(2, v)                                            # beam_inclinations, packed or not
        out[name]=dict(beam_inclinations=np.array(inc), inc_min=lo, inc_max=hi,
                       extrinsic=np.array(ext).reshape(4,4) if ext and len(ext)==16 else None)
    return out

if __name__=='__main__':
    raw=Path('/home/koyama/code/ST3D/data/waymo/raw_data')
    seq='segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
    cal=laser_calibrations(raw/seq)
    for k,v in cal.items():
        e=v['extrinsic']
        t=e[:3,3] if e is not None else None
        print(f"{k:11s} n_beam_inclinations={len(v['beam_inclinations']):3d}  "
              f"inc_min={v['inc_min']}  inc_max={v['inc_max']}  "
              f"translation={np.round(t,3) if t is not None else None}")
    top=cal['TOP']
    print('\nTOP extrinsic (lidar -> vehicle):\n', np.round(top['extrinsic'],4))
    bi=np.degrees(top['beam_inclinations'])
    if len(bi):
        bi=np.sort(bi)
        print(f"\nTOP beam inclinations: {len(bi)} beams, {bi.min():.2f}° .. {bi.max():.2f}°")
        print('  spacing min/median/max: %.3f / %.3f / %.3f deg'%tuple(np.percentile(np.diff(bi),[0,50,100])))
