"""Helper functions.

Variables:
- x, y : reference data.
- X, Y : test data.
- W    : number of samples in block for stats testing.
"""

import numpy as np
import jpype as jp
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib import rc

plt.rc('text', usetex=True)

jidt_dir = "/Users/cgoh0451/packages/infodynamics-dist-1.6.1"

if not jp.isJVMStarted():
    jp.startJVM(jp.getDefaultJVMPath(), "-ea", f"-Djava.class.path={jidt_dir}/infodynamics.jar", convertStrings=True)

def _check_length(x, n):
    if not x.shape[0] % n == 0:
        raise ValueError("Source not a multiple integer of window length.")

def _block_shuffle(x, W):
    shuffled_x = x.reshape(-1, W)
    order = np.arange(shuffled_x.shape[0])
    np.random.shuffle(order)
    return shuffled_x[order].reshape(-1)

def _calc_pvalue(null, non_perm):
    null = np.abs(null)
    non_perm = np.abs(non_perm)
    percentile = stats.percentileofscore(null, non_perm)
    pvalue = 1 - percentile / 100
    return pvalue

def calc_mi(x, y, local=False):
    x = jp.JArray(jp.JDouble, 1)(x.tolist())
    y = jp.JArray(jp.JDouble, 1)(y.tolist())
    calcClass = jp.JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
    calc = calcClass()
    calc.initialise()
    calc.setObservations(x, y)
    local_mi = calc.computeLocalOfPreviousObservations()
    if local:
        return np.array(local_mi)
    else:
        return np.mean(local_mi)

def calc_mi_and_do_stats(x, y, W):
    _check_length(x, W)
    non_perm = calc_mi(x, y)
    null = []
    for _ in range(200):
        shuffled_x = _block_shuffle(x, W)
        perm = calc_mi(shuffled_x, y)
        null.append(perm)
    pvalue = _calc_pvalue(null, non_perm)
    return non_perm, pvalue

def calc_mi_diff(x, y, X, Y):
    mi1 = calc_mi(x, y)
    mi2 = calc_mi(X, Y)
    return mi2 - mi1

def calc_mi_diff_and_do_stats(x, y, X, Y, W):
    _check_length(X, W)
    non_perm = calc_mi_diff(x, y, X, Y)
    i1 = calc_mi(x, y, local=True).reshape(W, -1).mean(axis=0)
    i2 = calc_mi(X, Y, local=True).reshape(W, -1).mean(axis=0)
    n1 = i1.shape[0]
    i = np.concatenate([i1, i2], axis=-1)
    null = []
    for _ in range(200):
        np.random.shuffle(i)
        perm = np.mean(i[n1:]) - np.mean(i[:n1])
        null.append(perm)
    pvalue = _calc_pvalue(null, non_perm)
    return non_perm, pvalue

def calc_cross_mi(x, y, X, Y, concat=False, local=False):
    if concat:
        N = X.shape[0]
        x = np.concatenate([x, X], axis=0)
        y = np.concatenate([y, Y], axis=0)
    x = jp.JArray(jp.JDouble, 1)(x.tolist())
    y = jp.JArray(jp.JDouble, 1)(y.tolist())
    calcClass = jp.JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
    calc = calcClass()
    calc.initialise()
    calc.setObservations(x, y)
    if concat:
        result = calc.computeLocalOfPreviousObservations()
        local_cross_mi = result[-N:]  # only keep the test data
    else:
        X = jp.JArray(jp.JDouble, 1)(X.tolist())
        Y = jp.JArray(jp.JDouble, 1)(Y.tolist())
        local_cross_mi = calc.computeLocalUsingPreviousObservations(X, Y)
    if local:
        return np.array(local_cross_mi)
    else:
        return np.mean(local_cross_mi)

def calc_cross_mi_and_do_stats(x, y, X, Y, W, concat=False):
    _check_length(X, W)
    non_perm = calc_cross_mi(x, y, X, Y, concat)
    null = []
    for _ in range(200):
        shuffled_X = _block_shuffle(X, W)
        perm = calc_cross_mi(x, y, shuffled_X, Y, concat=concat)
        null.append(perm)
    pvalue = _calc_pvalue(null, non_perm)
    return non_perm, pvalue

def calc_cross_mi_diff(x, y, X1, Y1, X2, Y2, concat):
    cmi1 = calc_cross_mi(x, y, X1, Y1, concat=concat)
    cmi2 = calc_cross_mi(x, y, X2, Y2, concat=concat)
    return cmi2 - cmi1

def calc_cross_mi_diff_and_do_stats(x, y, X1, Y1, X2, Y2, W, concat=False):
    _check_length(X1, W)
    _check_length(X2, W)
    non_perm = calc_cross_mi_diff(x, y, X1, Y1, X2, Y2, concat=concat)
    ci1 = calc_cross_mi(x, y, X1, Y1, concat=concat, local=True)
    ci2 = calc_cross_mi(x, y, X2, Y2, concat=concat, local=True)
    n1 = ci1.shape[0]
    ci = np.concatenate([ci1, ci2], axis=-1)
    null = []
    for _ in range(200):
        np.random.shuffle(ci)
        perm = np.mean(ci[n1:]) - np.mean(ci[:n1])
        null.append(perm)
    pvalue = _calc_pvalue(null, non_perm)
    return non_perm, pvalue

def calc_and_plot(x, y, X, Y, W, concat, filename):
    I1, p_I1 = calc_mi_and_do_stats(x, y, W)
    I2, p_I2 = calc_mi_and_do_stats(X, Y, W)
    DI, p_DI = calc_mi_diff_and_do_stats(x, y, X, Y, W)
    CI, p_CI = calc_cross_mi_and_do_stats(x, y, X, Y, W, concat)
    print(f"I1 {I1:.2f} {p_I1:.1g}")
    print(f"I2 {I2:.2f} {p_I2:.1g}")
    print(f"DI {DI:.2f} {p_DI:.1g}")
    print(f"CI {CI:.2f} {p_CI:.1g}")

    fig, ax = plt.subplots()
    ax.scatter(x, y, label="System 1 (Reference)")
    ax.scatter(X, Y, label="System 2 (Test)")
    ax.set_xlim(-4, 6)
    ax.set_ylim(-5, 4)
    ax.set_xlabel("$x$", fontsize=20)
    ax.set_ylabel("$y$", fontsize=20)
    ax.legend(loc=3, fontsize=16)
    ax.tick_params("both", labelsize=16)
    table = f"\\begin{{tabular}}{{ c c c }} & Value & $p$-value \\\\ \hline $I_1$ & {I1:.2f} & {p_I1:.1g} \\\\ $I_2$ & {I2:.2f} & {p_I2:.1g} \\\\  $\\Delta I_{{21}}$ & {DI:.2f} & {p_DI:.1g} \\\\ $CI_{{21}}$ & {CI:.2f} & {p_CI:.1g} \\end{{tabular}}"
    plt.text(2, -3, table, size=14)
    plt.tight_layout()
    print("Saving", filename)
    plt.savefig(filename)
    plt.close()

    print()

def calc(x, y, X, Y, W, concat):
    I1, p_I1 = calc_mi_and_do_stats(x, y, W)
    I2, p_I2 = calc_mi_and_do_stats(X, Y, W)
    DI, p_DI = calc_mi_diff_and_do_stats(x, y, X, Y, W)
    CI, p_CI = calc_cross_mi_and_do_stats(x, y, X, Y, W, concat)
    #print(f"I1 {I1:.2f} {p_I1:.1g}")
    #print(f"I2 {I2:.2f} {p_I2:.1g}")
    #print(f"DI {DI:.2f} {p_DI:.1g}")
    #print(f"CI {CI:.2f} {p_CI:.1g}")
    return I1, I2, DI, CI

def calc_and_plot_3(x, y, X1, Y1, X2, Y2, W, concat, filename):
    I1, p_I1 = calc_mi_and_do_stats(x, y, W)
    I2, p_I2 = calc_mi_and_do_stats(X1, Y1, W)
    I3, p_I3 = calc_mi_and_do_stats(X2, Y2, W)
    DI, p_DI = calc_mi_diff_and_do_stats(X1, Y1, X2, Y2, W)
    CI1, p_CI1 = calc_cross_mi_and_do_stats(x, y, X1, Y1, W, concat)
    CI2, p_CI2 = calc_cross_mi_and_do_stats(x, y, X2, Y2, W, concat)
    DCI, p_DCI = calc_cross_mi_diff_and_do_stats(x, y, X1, Y1, X2, Y2, W, concat)
    print(f"I1  {I1:.2f} {p_I1:.1g}")
    print(f"I2  {I2:.2f} {p_I2:.1g}")
    print(f"I3  {I3:.2f} {p_I3:.1g}")
    print(f"DI  {DI:.2f} {p_DI:.1g}")
    print(f"CI1 {CI1:.2f} {p_CI1:.1g}")
    print(f"CI2 {CI2:.2f} {p_CI2:.1g}")
    print(f"DCI {DCI:.2f} {p_DCI:.2g}")

    fig, ax = plt.subplots()
    ax.scatter(x, y, label="System 1 (Reference)")
    ax.scatter(X1, Y1, label="System 2 (Test)")
    ax.scatter(X2, Y2, label="System 3 (Test)")
    ax.set_xlim(-4, 6)
    ax.set_ylim(-6, 4)
    ax.set_xlabel("$x$", fontsize=20)
    ax.set_ylabel("$y$", fontsize=20)
    ax.legend(loc=3, fontsize=16)
    ax.tick_params("both", labelsize=16)
    table = f"\\begin{{tabular}}{{ c c c }} & Value & $p$-value \\\\ \hline $I_1$ & {I1:.2f} & {p_I1:.1g} \\\\ $I_2$ & {I2:.2f} & {p_I2:.1g} \\\\ $I_3$ & {I3:.2f} & {p_I3:.1g} \\\\  $\\Delta I_{{32}}$ & {DI:.2f} & {p_DI:.1g} \\\\ $CI_{{21}}$ & {CI1:.2f} & {p_CI1:.1g} \\\\ $CI_{{31}}$ & {CI2:.2f} & {p_CI2:.1g} \\\\ $\Delta CI_{{32}}$ & {DCI:.2f} & {p_DCI:.1g} \\end{{tabular}}"
    plt.text(2, -3.5, table, size=14)
    plt.tight_layout()
    print("Saving", filename)
    plt.savefig(filename)
    plt.close()

    print()

def calc_3(x, y, X1, Y1, X2, Y2, W, concat):
    I1, p_I1 = calc_mi_and_do_stats(x, y, W)
    I2, p_I2 = calc_mi_and_do_stats(X1, Y1, W)
    I3, p_I3 = calc_mi_and_do_stats(X2, Y2, W)
    DI, p_DI = calc_mi_diff_and_do_stats(X1, Y1, X2, Y2, W)
    CI1, p_CI1 = calc_cross_mi_and_do_stats(x, y, X1, Y1, W, concat)
    CI2, p_CI2 = calc_cross_mi_and_do_stats(x, y, X2, Y2, W, concat)
    DCI, p_DCI = calc_cross_mi_diff_and_do_stats(x, y, X1, Y1, X2, Y2, W, concat)
    #print(f"I1  {I1:.2f} {p_I1:.1g}")
    #print(f"I2  {I2:.2f} {p_I2:.1g}")
    #print(f"I3  {I3:.2f} {p_I3:.1g}")
    #print(f"DI  {DI:.2f} {p_DI:.1g}")
    #print(f"CI1 {CI1:.2f} {p_CI1:.1g}")
    #print(f"CI2 {CI2:.2f} {p_CI2:.1g}")
    #print(f"DCI {DCI:.2f} {p_DCI:.2g}")
    return I1, I2, I3, DI, CI1, CI2, DCI

def plot(x, y, X, Y, filename, legend=True):
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="System 1 (Reference)")
    ax.scatter(X, Y, label="System 2 (Test)")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-5, 4)
    ax.set_xlabel("$x$", fontsize=20)
    ax.set_ylabel("$y$", fontsize=20)
    if legend:
        ax.legend(loc=3, fontsize=16)
    ax.tick_params("both", labelsize=16)
    plt.tight_layout()
    print("Saving", filename)
    plt.savefig(filename)
    plt.close()
