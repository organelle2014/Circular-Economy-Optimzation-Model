import numpy as np
import matplotlib.pyplot as plt

from util import ConsumerMarket, MarketEquilibrium


def plot_utilities(market: ConsumerMarket, Ps: float, Pr: float, num=300, scenario_label=None):
    v = np.linspace(0, market.delta, num)
    uBuy = market.utility_buy(v, Ps)
    uRent = market.utility_rent(v, Pr)
    uOut = np.zeros_like(v)

    plt.figure(figsize=(8, 5))
    plt.plot(v, uBuy, label="Buy utility")
    plt.plot(v, uRent, label="Rent utility")
    plt.plot(v, uOut, label="Outside utility", linestyle="--")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.xlabel("Consumer type v")
    plt.ylabel("Utility")
    if scenario_label:
        plt.title(f"Utility by consumer type ({scenario_label})")
    else:
        plt.title("Utility by consumer type")
    plt.legend()
    plt.tight_layout()


def plot_market_shares(market: ConsumerMarket, Ps: float, Pr: float, scenario_label=None):
    shares = market.market_shares(Ps, Pr)
    labels = ["Buy", "Rent", "Outside"]
    values = [shares["buy"], shares["rent"], shares["outside"]]
    x = np.arange(len(labels))
    colors = ["#4C78A8", "#F58518", "#E45756"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        x,
        values,
        color="#999999",
        marker="o",
        linewidth=0.75,
        markeredgewidth=2,
        markeredgecolor="#000000",
        zorder=2,
    )
    for i in range(len(x) - 1):
        ax.plot(
            [x[i], x[i + 1]],
            [values[i], values[i + 1]],
            color=colors[i],
            linewidth=1.5,
            zorder=3,
        )
    for i, (xi, val) in enumerate(zip(x, values)):
        ax.scatter(
            [xi],
            [val],
            color=colors[i],
            s=50,
            edgecolor="#000000",
            linewidth=1.5,
            zorder=4,
        )
        ax.text(xi, val + 0.02, f"{val:.1%}", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Market share")
    if scenario_label:
        ax.set_title(f"Market shares ({scenario_label})")
    else:
        ax.set_title("Market shares")
    fig.tight_layout()


def plot_channel_comparison(eqIndependent, eqIntegrated):
    labels = ["Independent", "Integrated"]
    x = np.arange(len(labels))
    profitValues = [
        eqIndependent["profitSeller"] + eqIndependent["profitRenter"],
        eqIntegrated["profitTotal"],
    ]
    welfareValues = [eqIndependent["totalWelfare"], eqIntegrated["totalWelfare"]]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(
        x,
        profitValues,
        color="#4C78A8",
        marker="o",
        linewidth=1.25,
        markeredgewidth=2,
        markeredgecolor="#000000",
    )
    ax[0].set_title("Total profit (Independent vs Integrated)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    for i, val in enumerate(profitValues):
        ax[0].text(i, val, f"{val:.2f}", ha="center", va="bottom")

    ax[1].plot(
        x,
        welfareValues,
        color="#F58518",
        marker="o",
        linewidth=1.25,
        markeredgewidth=2,
        markeredgecolor="#000000",
    )
    ax[1].set_title("Total welfare (Independent vs Integrated)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    for i, val in enumerate(welfareValues):
        ax[1].text(i, val, f"{val:.2f}", ha="center", va="bottom")

    fig.tight_layout()


def main():
    market = ConsumerMarket(delta=100, k=5, X=0.5, w=10, q=0.8)

    # Example prices for utilities and shares
    Ps, Pr = 35, 20
    plot_utilities(market, Ps, Pr, scenario_label="Example prices")
    plot_market_shares(market, Ps, Pr, scenario_label="Example prices")

    # Channel comparison
    eqSolver = MarketEquilibrium(market)
    eqIndependent = eqSolver.monopoly_equilibrium(mode="sequential")
    eqIntegrated = eqSolver.integrated_channel()
    plot_channel_comparison(eqIndependent, eqIntegrated)

    plt.show()


if __name__ == "__main__":
    main()
