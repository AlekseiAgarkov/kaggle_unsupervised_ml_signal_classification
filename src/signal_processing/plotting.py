import numpy as np
from matplotlib import pyplot as plt

from src.signal_processing.metrics import eval_metrics

small_font = dict(fontsize=10)
wide_plt = dict(figsize=(12, 3), legend=None)
x_label = "Time, ns"
y_label = "Bit ADC"
thin = dict(alpha=0.25, linewidth=0.25)
highlight = dict(alpha=0.8, linewidth=1.5)
basic_line = dict(**thin, color='lightslategrey')
highlight_line = dict(**highlight, color='salmon')
signals_title = dict(title='Signals')
red_dash = dict(color='r', linestyle='--', linewidth=0.5)
split_line = dict(color='blue', linestyle='--', linewidth=0.5)
markers = dict(s=0.5, alpha=0.25)


def set_axis_labels(ax):
    """
    Именование осей
    """
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def plot_plain_signal(signal, title, xlim=None, ylim=None, **kwargs):
    """
    Визуализация сигналов.
    """
    ax = signal.plot(title=title, **wide_plt, **kwargs)
    ax.set_xlabel("Time, ns")
    ax.set_ylabel("Bit ADC")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    return ax


def ax_off(ax):
    """
    Отключение осей
    """
    ax.axis("off")


def plot_hist(df, col, bins, ax, title, ymax=None, ax_zoom=None):
    """
    Базовая гистограмма
    """
    ax.hist(df[col], bins=bins)
    ax.set_title(title, **small_font)
    if ymax is not None and ax_zoom is not None:
        ax_zoom.hist(df[col], bins=bins)
        ax_zoom.set_title(title + " (низкочастотные)", **small_font)
        ax_zoom.set_ylim((0, ymax))


def plot_corr_m(df):
    """
    Матрица корреляций признаков
    """
    components = df.columns.tolist()

    plt.imshow(df[components].corr(), aspect='auto')
    plt.colorbar(label='Коэффициент корреляции')

    plt.xticks(range(len(components)), components, rotation=30)
    plt.yticks(range(len(components)), components)
    plt.title('Корреляционная матрица компонентов сигнала', fontsize=14)

    plt.show()


def plot_eval_hist(arr, title, x_label, bins=100):
    """
    Вывод гистограммы с наложением GaussianMixture и границы разделения кластеров.
    """
    metrics = eval_metrics(arr, bins=bins)

    counts = metrics["counts"]
    bin_arr = metrics["bin_arr"]
    gm_x = metrics["gm_x"]
    gm_y = metrics["gm_y"]
    valley = metrics["valley"]

    fom = metrics["fom"]
    sil = metrics["silhouette_score"]
    ch = metrics["calinski_harabasz_score"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(bin_arr[:-1], counts, width=np.diff(bin_arr), alpha=0.5, color='b', edgecolor='black', align='edge')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Частота')
    ax2 = ax.twinx()
    ax2.plot(gm_x, gm_y, **highlight_line)
    ax2.set_ylim(bottom=0)

    ax.axvline(valley, **highlight_line)

    plt.title(title + f'\n(FOM={fom:.3f}, CH={ch:.3f}, Silhouette={sil:.3f})')
    return metrics
