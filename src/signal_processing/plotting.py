from matplotlib import pyplot as plt

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


def set_axis_labels(ax):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def plot_plain_signal(signal, title, xlim=None, ylim=None, **kwargs):
    ax = signal.plot(title=title, **wide_plt, **kwargs)
    ax.set_xlabel("Time, ns")
    ax.set_ylabel("Bit ADC")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    return ax


def ax_off(ax):
    ax.axis("off")


def plot_hist(df, col, bins, ax, title, ymax=None, ax_zoom=None):
    ax.hist(df[col], bins=bins)
    ax.set_title(title, **small_font)
    if ymax is not None and ax_zoom is not None:
        ax_zoom.hist(df[col], bins=bins)
        ax_zoom.set_title(title + " (низкочастотные)", **small_font)
        ax_zoom.set_ylim((0, ymax))


def plot_corr_m(df):
    components = df.columns.tolist()
    plt.imshow(df[components].corr(), aspect='auto')
    plt.colorbar(label='Коэффициент корреляции')
    plt.xticks(range(len(components)), components, rotation=30)
    plt.yticks(range(len(components)), components)
    plt.title('Корреляционная матрица компонентов сигнала', fontsize=14)
    plt.show()
