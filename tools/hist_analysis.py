import os
import numpy as np
import scipy.stats as stats

def main():
    version = "517oe15r"
    file_paths = {
        "lyft": f"/storage/hist_dist_lyft_{version}.npy",
        "nuscenes": f"/storage/hist_dist_nuscenes_{version}.npy",
        "kitti": f"/storage/hist_dist_kitti_{version}.npy",
        "pandaset": f"/storage/hist_dist_pandaset_{version}.npy",
        "waymo": f"/storage/hist_dist_waymo_{version}.npy"
    }

    histograms = {}
    for key, path in file_paths.items():
        if os.path.exists(path):
            histograms[key] = np.load(path)
        else:
            raise FileNotFoundError(f"File not found: {path}")

    hist_lyft = histograms["lyft"].flatten()
    hist_nuscenes = histograms["nuscenes"].flatten()
    hist_kitti = histograms["kitti"].flatten()
    hist_pandaset = histograms["pandaset"].flatten()
    hist_waymo = histograms["waymo"].flatten()

    histograms_kitti_calibrated = {}
    for key, hist in histograms.items():
        hist_calibrated = [0] * len(hist)
        for i in range(len(hist)):
            if hist[i] > hist_kitti[i]:
                hist_calibrated[i] = hist_kitti[i]
            else:
                hist_calibrated[i] = hist[i]
        histograms_kitti_calibrated[key] = np.array(hist_calibrated)

    histograms_nuscenes_calibrated = {}
    for key, hist in histograms.items():
        hist_calibrated = [0] * len(hist)
        for i in range(len(hist)):
            if hist[i] > hist_nuscenes[i]:
                hist_calibrated[i] = hist_nuscenes[i]
            else:
                hist_calibrated[i] = hist[i]
        histograms_nuscenes_calibrated[key] = np.array(hist_calibrated)

    ref_distance = stats.wasserstein_distance(hist_kitti, hist_lyft)

    print("wasserstain distance between kitti and lyft: ", stats.wasserstein_distance(hist_kitti, hist_lyft) / ref_distance)
    print("wasserstain distance between kitti and nuscenes: ", stats.wasserstein_distance(hist_kitti, hist_nuscenes) / ref_distance)
    print("wasserstain distance between kitti and pandaset: ", stats.wasserstein_distance(hist_kitti, hist_pandaset) / ref_distance)
    print("wasserstain distance between kitti and waymo: ", stats.wasserstein_distance(hist_kitti, hist_waymo)/ ref_distance)
    print("===========================")
    print("wasserstain distance between kitti and kitti_calibrated_lyft: ", stats.wasserstein_distance(hist_kitti, histograms_kitti_calibrated["lyft"]) / ref_distance)
    print("wasserstain distance between kitti and kitti_calibrated_nuscenes: ", stats.wasserstein_distance(hist_kitti, histograms_kitti_calibrated["nuscenes"]) / ref_distance)
    print("wasserstain distance between kitti and kitti_calibrated_pandaset: ", stats.wasserstein_distance(hist_kitti, histograms_kitti_calibrated["pandaset"]) / ref_distance)
    print("wasserstain distance between kitti and kitti_calibrated_waymo: ", stats.wasserstein_distance(hist_kitti, histograms_kitti_calibrated["waymo"]) / ref_distance)
    print("===========================")
    print("wasserstain distance between nuscenes and lyft: ", stats.wasserstein_distance(hist_nuscenes, hist_lyft)/ ref_distance)
    print("wasserstain distance between nuscenes and kitti: ", stats.wasserstein_distance(hist_nuscenes, hist_kitti)/ ref_distance)
    print("wasserstain distance between nuscenes and pandaset: ", stats.wasserstein_distance(hist_nuscenes, hist_pandaset)/ ref_distance)
    print("wasserstain distance between nuscenes and waymo: ", stats.wasserstein_distance(hist_nuscenes, hist_waymo)/ ref_distance)
    print("===========================")
    print("wasserstain distance between nuscenes and nuscenes_calibrated_lyft: ", stats.wasserstein_distance(hist_nuscenes, histograms_nuscenes_calibrated["lyft"])/ ref_distance)
    print("wasserstain distance between nuscenes and nuscenes_calibrated_waymo: ", stats.wasserstein_distance(hist_nuscenes, histograms_nuscenes_calibrated["waymo"])/ ref_distance)
    print("wasserstain distance between nuscenes and nuscenes_calibrated_pandaset: ", stats.wasserstein_distance(hist_nuscenes, histograms_nuscenes_calibrated["pandaset"])/ ref_distance)
    print("wasserstain distance between nuscenes and nuscenes_calibrated_kitti: ", stats.wasserstein_distance(hist_nuscenes, histograms_nuscenes_calibrated["kitti"])/ ref_distance)

if __name__ == '__main__':
    main()
