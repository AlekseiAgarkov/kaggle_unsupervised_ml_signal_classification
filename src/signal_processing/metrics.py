import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture


def eval_metrics(arr, bins=100):
    """
    Расчёт метрик Figure of Merit, Calinski-Harabasz Score, Silhouette Score.
    """
    arr = arr[~np.isnan(arr) & ~np.isinf(arr)]
    counts, bin_arr = np.histogram(arr, bins=bins)

    peaks, _ = find_peaks(counts)
    top2_peaks = np.sort(peaks[np.argsort(counts[peaks])[-2:]])
    valley = bin_arr[np.argmin(counts[top2_peaks[0]:top2_peaks[1] + 1]) + top2_peaks[0]]

    gm_model = GaussianMixture(
        n_components=2,
        max_iter=1000,
        random_state=42,
        init_params='kmeans',
        tol=1e-4,
        n_init=10
    )

    gm_model.fit(arr.reshape(-1, 1))

    mu = gm_model.means_.flatten()
    sigma = np.sqrt(gm_model.covariances_.flatten())
    fom = abs((mu[1] - mu[0]) / (2.355 * (sigma[1] + sigma[0])))

    x = np.linspace(np.min(arr), np.max(arr), 1000)
    y = np.exp(gm_model.score_samples(x.reshape(-1, 1)))

    labels = (arr > valley).astype(int)
    sil_score = silhouette_score(arr.reshape(-1, 1), labels)

    ch_score = calinski_harabasz_score(arr.reshape(-1, 1), labels)

    return {"valley": valley,
            "counts": counts,
            "bin_arr": bin_arr,
            "gm_x": x,
            "gm_y": y,
            "mu": mu,
            "sigma": sigma,
            "fom": fom,
            "silhouette_score": sil_score,
            "calinski_harabasz_score": ch_score}
