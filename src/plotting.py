import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def setup_plot_style(cfg):
    """Setzt globalen Plot-Stil aus der Config."""
    style = cfg.get("plot_style", {})
    sns.set_style(style.get("style", "whitegrid"))
    plt.rcParams["figure.dpi"] = style.get("dpi", 100)
    sns.set_palette(style.get("palette", "Set2"))


def plot_comparison_bars(crosstab_df, title, xlabel, ylabel, colors=None, stacked=False, figsize=(14, 7)):
    """Erstellt ein gruppiertes oder gestapeltes Balkendiagramm aus einer Kreuztabelle."""
    if colors is None:
        colors = ["red", "green"]
    plt.figure(figsize=figsize)
    crosstab_df.plot(kind="bar", stacked=stacked, color=colors)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Label")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_wordclouds(texts, colormaps, titles, figsize=(15, 6)):
    """Erstellt nebeneinander liegende Wordclouds.

    Args:
        texts: Liste von Text-Strings
        colormaps: Liste von Colormap-Namen (z.B. ['Reds', 'Greens'])
        titles: Liste von Titeln
    """
    n = len(texts)
    plt.figure(figsize=figsize)
    for i, (text, cmap, title) in enumerate(zip(texts, colormaps, titles)):
        plt.subplot(1, n, i + 1)
        wc = WordCloud(background_color="white", max_words=1000, colormap=cmap, width=800, height=400).generate(text)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()
