"""How much does the FIRST sparse convolution's output spread change between datasets, per encoder?

experiments_md/20261005_01. The first layer of VoxelResBackBone8x is a submanifold 3x3x3 conv: each
occupied voxel sums W_k @ feature over its occupied neighbours. KITTI has about twice nuScenes'
occupied neighbours per voxel (5.2 vs 2.6), so the response spread depends on how features add up:
  - MeanVFE feeds ABSOLUTE voxel-mean xyz (tens of metres, same sign in a neighbourhood): the terms
    add coherently and the spread grows ~linearly with neighbour count;
  - GBlobs' position block feeds the voxel mean MINUS the voxel centre (+-5 cm, roughly zero-mean):
    the terms add incoherently, ~sqrt(count), and nothing depends on where the voxel is.
This proxy uses RANDOM weights (one fixed draw), so it measures the encoding, not a trained model; the
trained models' first BatchNorm gap is in each adabn_eval.py run's bn_gap.json. Raw val files,
SHIFT_COOR applied (nuScenes 1.75, KITTI 1.70), POINT_CLOUD_RANGE and VOXEL_SIZE of da-ieee-access.

    python analysis/first_layer_density_proxy.py      # in the container, ~2 min, CPU
"""
import pickle, numpy as np
ROOT='/home/koyama/code/ST3D/data/'
VS=np.array([0.1,0.1,0.15]); PCR=np.array([-75.2,-75.2,-2,75.2,75.2,4])
rng=np.random.default_rng(0); W=rng.normal(size=(27,3,16))
OFF=np.array([(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)])
def voxelize(p):
    m=np.all((p>=PCR[:3])&(p<PCR[3:]),1); p=p[m]
    ijk=np.floor((p-PCR[:3])/VS).astype(np.int64)
    key=(ijk[:,0]*2000+ijk[:,1])*100+ijk[:,2]
    u,inv,cnt=np.unique(key,return_inverse=True,return_counts=True)
    mean=np.zeros((len(u),3)); np.add.at(mean,inv,p); mean/=cnt[:,None]
    ijk_u=np.zeros((len(u),3),np.int64); ijk_u[inv]=ijk
    centre=PCR[:3]+(ijk_u+0.5)*VS
    return ijk_u,mean,mean-centre
def subm(ijk,feat):
    key=lambda a:(a[:,0]*2000+a[:,1])*100+a[:,2]
    k0=key(ijk); order=np.argsort(k0); ks=k0[order]
    out=np.zeros((len(ijk),16)); nn=np.zeros(len(ijk))
    for o,w in zip(OFF,W):
        q=key(ijk+o); pos=np.searchsorted(ks,q); pos=np.clip(pos,0,len(ks)-1); hit=ks[pos]==q
        idx=order[pos[hit]]; out[hit]+=feat[idx]@w; nn+=hit
    return out,nn-1
def frames(ds,n):
    if ds=='nuScenes':
        infos=pickle.load(open(ROOT+'nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl','rb'))
        for i in np.linspace(0,len(infos)-1,n).astype(int):
            p=np.fromfile(ROOT+'nuscenes/v1.0-trainval/'+infos[i]['lidar_path'],np.float32).reshape(-1,5)[:,:3].astype(float)
            p=p[~((abs(p[:,0])<1.5)&(abs(p[:,1])<1.5))]; p[:,2]+=1.75; yield p
    else:
        infos=pickle.load(open(ROOT+'kitti/kitti_infos_val.pkl','rb'))
        for i in np.linspace(0,len(infos)-1,n).astype(int):
            p=np.fromfile(ROOT+f"kitti/training/velodyne/{infos[i]['point_cloud']['lidar_idx']}.bin",np.float32).reshape(-1,4)[:,:3].astype(float)
            p[:,2]+=1.70; yield p
res={}
for ds in ['nuScenes','KITTI']:
    A=[];R=[];N=[];V=[]
    for p in frames(ds,40):
        ijk,mean,rel=voxelize(p)
        a,nn=subm(ijk,mean); r,_=subm(ijk,rel)
        A.append(a);R.append(r);N.append(nn);V.append(len(ijk))
    A=np.concatenate(A);R=np.concatenate(R);N=np.concatenate(N)
    res[ds]=(A.std(0),R.std(0),A.mean(0),R.mean(0))
    print(f"{ds:9s} occupied voxels/frame {np.mean(V):8.0f}   occupied neighbours per voxel: mean {N.mean():.2f}, median {np.median(N):.0f}")
for name,i in [('MeanVFE input (absolute xyz)',0),('GBlobs position block (offset)',1)]:
    sr=res['KITTI'][i]/res['nuScenes'][i]
    ms=np.abs(res['KITTI'][i+2]-res['nuScenes'][i+2])/res['nuScenes'][i]
    print(f"{name:32s} first-layer spread KITTI/nuScenes: median {np.median(sr):.2f} [range {sr.min():.2f}-{sr.max():.2f}], |log sd ratio| median {np.median(abs(np.log(sr))):.3f}; mean shift median {np.median(ms):.3f} sd")
