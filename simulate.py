"""Generate simulation results.

"""

import numpy as np
from functions import calc_and_plot, calc_and_plot_3, calc, plot

plot_fig1 = True
plot_fig2 = True
plot_fig3 = True
plot_fig4 = True
plot_fig5 = True
plot_fig6 = True
plot_fig7 = True

N = np.random.normal
Unif = np.random.uniform

if plot_fig1:
    n = 2000  # Number of baseline samples
    m = 200   # Number of test samples
    W = 1     # Block length for stats testing

    X = N(1.5, 0.1, size=m); Y = N(0.5*X, 0.1, size=m)

    x = N(size=n); y = N(size=n)
    calc_and_plot(x, y, X, Y, W=W, concat=False, filename="plots/fig1a.png")

    x = N(0, 0.2, size=n); y = N(0, 0.2, size=n)
    calc_and_plot(x, y, X, Y, W=W, concat=False, filename="plots/fig1b.png")

    x = N(size=n); y = N(0.5*x, 0.1, size=n)
    calc_and_plot(x, y, X, Y, W=W, concat=False, filename="plots/fig1c.png")

    x = N(0, 0.2, size=n); y = N(0.5*x, 0.1, size=n)
    calc_and_plot(x, y, X, Y, W=W, concat=False, filename="plots/fig1d.png")

if plot_fig2:
    n = 1000  # Number of baseline samples
    m = 1000   # Number of test samples
    W = 1     # Block length for stats testing

    X = N(1, 0.25, size=m); Y = N(0.5*X, 0.05, size=m)

    # Redundancy
    x = N(size=n); y = N(0.5*x, 0.05, size=n)
    calc_and_plot(x, y, X, Y, W=W, concat=False, filename="plots/fig2a.png")

    # Synergy
    x = N(size=n); y = N(0.5*x, 0.5, size=n)
    calc_and_plot(x, y, X, Y, W=W, concat=False, filename="plots/fig2b.png")

if plot_fig3:
    n = 2000  # Number of baseline samples
    m = 200   # Number of test samples
    W = 1     # Block length for stats testing

    p = Unif(0, 2*np.pi)

    x = N(size=n); y = N(0.5*x, 0.1, size=n)
    X = N(1, 0.5, size=m); Y = N(np.sin(2*np.pi*X + p) + 1, 0.05, size=m)

    calc_and_plot(x, y, X, Y, W=W, concat=False, filename="plots/fig3.png")

if plot_fig4:
    n = 2000  # Number of baseline samples
    m = 200   # Number of test samples
    W = 1     # Block length for stats testing

    X = N(1.5, 0.1, size=m); Y = N(0.5*X, 0.1, size=m)

    x = N(size=n); y = N(size=n)
    calc_and_plot(x, y, X, Y, W=W, concat=True, filename="plots/fig4a.png")

    x = N(0, 0.2, size=n); y = N(0, 0.2, size=n)
    calc_and_plot(x, y, X, Y, W=W, concat=True, filename="plots/fig4b.png")

    x = N(size=n); y = N(0.5*x, 0.1, size=n)
    calc_and_plot(x, y, X, Y, W=W, concat=True, filename="plots/fig4c.png")

    x = N(0, 0.2, size=n); y = N(0.5*x, 0.1, size=n)
    calc_and_plot(x, y, X, Y, W=W, concat=True, filename="plots/fig4d.png")

if plot_fig5:
    import matplotlib.pyplot as plt
    from matplotlib import rc

    plt.rc('text', usetex=True)

    def plot_scatter(M, I1, I2, DI, CI, filename):
        fig, ax = plt.subplots()
        ax.scatter(M, I1, label="$I_1$")
        ax.plot(M, I1, linestyle="--")
        ax.scatter(M, I2, label="$I_2$")
        ax.plot(M, I2, linestyle="--")
        ax.scatter(M, DI, label="$\Delta I_{21}$")
        ax.plot(M, DI, linestyle="--")
        ax.scatter(M, CI, label="$CI_{21}$")
        ax.plot(M, CI, linestyle="--")
        ax.set_xlabel("Number of test data samples", fontsize=18)
        ax.set_ylabel("Measure (nats)", fontsize=18)
        ax.legend(loc=3, fontsize=16)
        ax.tick_params("both", labelsize=16)
        plt.tight_layout()
        print("Saving", filename)
        plt.savefig(filename)
        plt.close()

    x = N(size=2000); y = N(size=2000)
    X = N(1.5, 0.4, size=200); Y = N(0.5*X, 0.1, size=200)
    plot(x, y, X, Y, filename="plots/fig5ai.png")

    M = [50, 100, 200, 500, 1000]
    I1, I2, DI, CI = [], [], [], []
    for m in M:
        X = N(1.5, 0.4, size=m)
        Y = N(X, 0.1, size=m)

        i1, i2, di, ci = calc(x, y, X, Y, W=1, concat=True)
        print(m, i1, i2, di, ci)
        I1.append(i1)
        I2.append(i2)
        DI.append(di)
        CI.append(ci)

    plot_scatter(M, I1, I2, DI, CI, filename="plots/fig5aii.png")

    p = Unif(0, 2*np.pi)
    X = N(1, 0.5, size=200); Y = N(np.sin(2*np.pi*X + p) + 1, 0.05, size=200)
    plot(x, y, X, Y, filename="plots/fig5bi.png")

    I1, I2, DI, CI = [], [], [], []
    for m in M:
        X = N(1, 0.5, size=m)
        Y = N(np.sin(2*np.pi*X + p) + 1, 0.05, size=m)

        i1, i2, di, ci = calc(x, y, X, Y, W=1, concat=True)
        print(m, i1, i2, di, ci)
        I1.append(i1)
        I2.append(i2)
        DI.append(di)
        CI.append(ci)

    plot_scatter(M, I1, I2, DI, CI, filename="plots/fig5bii.png")

    x = N(size=2000); y = N(0.5*x, 0.1, size=2000)
    X = N(1.5, 0.4, size=200); Y = N(0.5*X, 0.1, size=200)
    plot(x, y, X, Y, filename="plots/fig5ci.png")

    M = [50, 100, 200, 500, 1000]
    I1, I2, DI, CI = [], [], [], []
    for m in M:
        X = N(1.5, 0.4, size=m)
        Y = N(X, 0.1, size=m)

        i1, i2, di, ci = calc(x, y, X, Y, W=1, concat=True)
        print(m, i1, i2, di, ci)
        I1.append(i1)
        I2.append(i2)
        DI.append(di)
        CI.append(ci)

    plot_scatter(M, I1, I2, DI, CI, filename="plots/fig5cii.png")

    p = Unif(0, 2*np.pi)
    X = N(1, 0.5, size=200); Y = N(np.sin(2*np.pi*X + p) + 1, 0.05, size=200)
    plot(x, y, X, Y, filename="plots/fig5di.png")

    I1, I2, DI, CI = [], [], [], []
    for m in M:
        X = N(1, 0.5, size=m)
        Y = N(np.sin(2*np.pi*X + p) + 1, 0.05, size=m)

        i1, i2, di, ci = calc(x, y, X, Y, W=1, concat=True)
        print(m, i1, i2, di, ci)
        I1.append(i1)
        I2.append(i2)
        DI.append(di)
        CI.append(ci)

    plot_scatter(M, I1, I2, DI, CI, filename="plots/fig5dii.png")

if plot_fig6:
    n = 200
    m = 200
    W = 1

    x = [
        N(-2, 0.2, size=n),
        N(2, 0.2, size=n),
        N(-2, 0.2, size=n),
        N(2, 0.2, size=n),
    ]
    y = [
        N(-0.5*(x[0]+2) + 3, 0.1, size=n),
        N(0.5*(x[1]-2) + 3, 0.1, size=n),
        N(0.5*(x[2]+2) - 1, 0.1, size=n),
        N(-0.5*(x[3]-2) - 1, 0.1, size=n),
    ]
    x = np.concatenate(x)
    y = np.concatenate(y)

    X = N(2, 0.2, size=m)
    Y = N(0.5*(X-2) + 3, 0.1, size=m)
    calc_and_plot(x, y, X, Y, W=W, concat=True, filename="plots/fig6a.png")

    p = Unif(0, 2*np.pi)
    X = N(2, 0.2, size=m)
    Y = N(1.2*X + 0.75, 0.05, size=m)
    calc_and_plot(x, y, X, Y, W=W, concat=True, filename="plots/fig6b.png")

    X2 = N(-2, 0.2, size=m)
    Y2 = N(1.2*X2 + 1.5, 0.05, size=m)
    calc_and_plot_3(x, y, X, Y, X2, Y2, W=W, concat=True, filename="plots/fig6c.png")

    p = Unif(0, 2*np.pi)
    Y = N(0.5*np.sin(2*np.pi*X + p) + 3, 0.2, size=m)
    calc_and_plot_3(x, y, X, Y, X2, Y2, W=W, concat=True, filename="plots/fig6d.png")

if plot_fig7:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from statsmodels.graphics import tsaplots
    from cycler import cycler

    n = 100
    ex = N(0, 0.05, n)
    ey = N(0, 0.05, n)
    t = np.arange(n)
    x = np.zeros(n)
    y = np.zeros(n)
    for t_ in t[1:]:
        x[t_] = 0.8 * x[t_ - 1] + ex[t_]
        y[t_] = 0.3 * y[t_ - 1] + 0.9 * x[t_ - 1] + ey[t_]

    ex = N(0, 0.05, n)
    ey = N(0, 0.05, n)
    t = np.arange(n)
    X = np.zeros(n)
    Y = np.zeros(n)
    for t_ in t[1:]:
        X[t_] = 0.8 * X[t_ - 1] + ex[t_]
        Y[t_] = 0.9 * X[t_ - 1] + ey[t_]

    calc_and_plot(x, y, X, Y, W=5, concat=True, filename="plots/fig7a.png")

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
    ax[0].scatter(t, x)
    ax[0].plot(t, x, linestyle="--")
    ax[0].set_title("Data")
    ax[0].set_xlabel("$t$", fontsize=14)
    ax[0].set_ylabel("$x$", fontsize=14)
    ax[0].tick_params("both", labelsize=13)
    tsaplots.plot_acf(x, ax=ax[1])
    ax[1].set_xlabel("Lag", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/fig7bi.png")
    plt.close()

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
    ax[0].scatter(t, y)
    ax[0].plot(t, y, linestyle="--")
    ax[0].set_title("Data")
    ax[0].set_xlabel("$t$", fontsize=14)
    ax[0].set_ylabel("$y$", fontsize=14)
    ax[0].tick_params("both", labelsize=13)
    tsaplots.plot_acf(y, ax=ax[1])
    ax[1].set_xlabel("Lag", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/fig7bii.png")


    mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:orange'])

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
    ax[0].scatter(t, X)
    ax[0].plot(t, X, linestyle="--")
    ax[0].set_title("Data")
    ax[0].set_xlabel("$t$", fontsize=14)
    ax[0].set_ylabel("$x$", fontsize=14)
    ax[0].tick_params("both", labelsize=13)
    tsaplots.plot_acf(X, ax=ax[1])
    ax[1].set_xlabel("Lag", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/fig7ci.png")
    plt.close()

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
    ax[0].scatter(t, Y)
    ax[0].plot(t, Y, linestyle="--")
    ax[0].set_title("Data")
    ax[0].set_xlabel("$t$", fontsize=14)
    ax[0].set_ylabel("$y$", fontsize=14)
    ax[0].tick_params("both", labelsize=13)
    tsaplots.plot_acf(Y, ax=ax[1])
    ax[1].set_xlabel("Lag", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/fig7cii.png")

