import argparse
import numpy as np
import matplotlib as mpl
import scipy.stats as stats
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.ticker import StrMethodFormatter
from matplotlib.colors import LinearSegmentedColormap

parser = argparse.ArgumentParser(description="Plotting tool", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--type", default="interpretation", type=str, help="Type of problem to plot")
parser.add_argument("--path", type=Path, action="append", help="Path of file to plot")

mpl.rc("font", family = "Times New Roman", size = 12)

def main(args):
    if (args.type == "interpretation"):
        interpretation("frequentist", 0, 16)
        interpretation("bayesian", 0, 16)

    elif (args.type == "inference"):
        inference(0, 10, 5, 5, 1)

    elif (args.type == "divergence"):
        divergence(0, 4, 2, 4)

    elif (args.type == "evidence"):
        evidence()

    elif (args.type == "activation"):
        activation("rectified")
        activation("sigmoid")
        activation("hyperbolic")

    elif (args.type == "optimisation"):
        optimisation()

    elif (args.type == "network"):
        network(1/12, [2, 4, 1], ["X", "H", "Y"], ["orange", "red", "magenta"])

    elif (args.type == "results"):
        results(args.path)
    
    elif (args.type == "logs"):
        logs(args.path, "evidence")
        logs(args.path, "loss")
        logs(args.path, "accuracy")

def hex_to_rgb(value):
    value = value.strip("#")
    return [int(value[i:i + 2], 16) for i in range(0, 6, 2)]

def rgb_to_dec(value):
    return [v / 256 for v in value]

def continuous_cmap(hex_list, float_list = None):
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    
    if (float_list == None):
        float_list = list(np.linspace(0, 1, len(rgb_list)))
        
    col_dict = {}
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i, _ in enumerate(float_list)]
        col_dict[col] = col_list
    col_map = LinearSegmentedColormap("col_map", segmentdata = col_dict, N = 256)
    return col_map

def interpretation(interpretation, mean, variance):
    fig = plt.figure(figsize = (7, 5))
    ax = fig.gca()
    
    x = np.linspace(-10, 10, 1000)
    y = stats.norm.pdf(x, mean, np.sqrt(variance))
    
    if (interpretation == "frequentist"):
        ax.plot(x, y, color = "orange", linewidth = 2.0, zorder = 2)
        ax.scatter([0, 1, -1, 3, -3, 6, -6, 10, -10], np.zeros(9), marker = "x", color = "black", linewidth = 2.0, zorder = 2)

    if (interpretation == "bayesian"):
        ax.plot(x, y, color = "red", linewidth = 2.0, zorder = 2)
        ax.scatter([0], np.zeros(1), marker = "x", color = "black", linewidth = 2.0, zorder = 2)
    
    plt.xticks([-10,-8,-6,-4,-2,0,2,4,6,8,10], [-10,-8,-6,-4,-2,0,2,4,6,8,10])
    plt.yticks([0.00,0.02,0.04,0.06,0.08,0.10], [0.00,0.02,0.04,0.06,0.08,0.10])
    
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:d}"))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))

    plt.xlim(-10, 10)
    plt.ylim(-0.01, 0.10)

    plt.xlabel("X", fontsize = 16)
    plt.ylabel("Probability", fontsize = 16)

    plt.grid()
    plt.show()

def inference(prior_mean, prior_variance, likelihood_mean, likelihood_variance, data_points):
    fig = plt.figure(figsize = (7, 5))
    ax = fig.gca()

    posterior_variance = 1 / (1 / prior_variance + data_points / likelihood_variance)
    posterior_mean = posterior_variance * (prior_mean / prior_variance + data_points * likelihood_mean / likelihood_variance)

    x = np.linspace(-10, 10, 1000)
    prior = stats.norm.pdf(x, prior_mean, np.sqrt(prior_variance))
    likelihood = stats.norm.pdf(x, likelihood_mean, np.sqrt(likelihood_variance))
    posterior = stats.norm.pdf(x, posterior_mean, np.sqrt(posterior_variance))

    ax.plot(x, prior, color = "orange", linewidth = 2.0, label = "Prior", zorder = 2)
    ax.plot(x, posterior, color = "red", linewidth = 2.0, label = "Posterior", zorder = 2)
    ax.plot(x, likelihood, color = "magenta", linewidth = 2.0, label = "Likelihood", zorder = 2)
    ax.legend(loc = "upper left", fontsize = 16, framealpha = 0)

    plt.xticks([-10.0,-7.5,-5.0,-2.5,0.0,2.5,5.0,7.5,10.0], [-10.0,-7.5,-5.0,-2.5,0.0,2.5,5.0,7.5,10.0])
    plt.yticks([0.00,0.05,0.10,0.15,0.20,0.25], [0.00,0.05,0.10,0.15,0.20,0.25])
    
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    
    plt.xlim(-10, 10)
    plt.ylim(0.00, 0.25)
    
    plt.xlabel("X", fontsize = 16)
    plt.ylabel("Probability", fontsize = 16)
    
    plt.grid()
    plt.show()

def divergence(q_mean, q_variance, p_mean, p_variance):
    fig = plt.figure(figsize = (7, 5))
    ax = fig.gca()
    
    x = np.linspace(-10, 10, 1000)
    q = stats.norm.pdf(x, q_mean, np.sqrt(q_variance))
    p = stats.norm.pdf(x, p_mean, np.sqrt(p_variance))
    kl = q * np.log(q / p)

    ax.fill_between(x, 0, kl, color = "orange", alpha = 1, zorder = 2)
    ax.plot(x, kl, color = "orange", linewidth = 2.0, label = "KL(Q||P)", zorder = 2)
    ax.plot(x, q, color = "red", linewidth = 2.0, label = "Q(X)", zorder = 2)
    ax.plot(x, p, color = "magenta", linewidth = 2.0, label = "P(X)", zorder = 2)
    ax.legend(loc = "upper left", fontsize = 16, framealpha = 0)

    plt.xticks([-10.0,-7.5,-5.0,-2.5,0.0,2.5,5.0,7.5,10.0], [-10.0,-7.5,-5.0,-2.5,0.0,2.5,5.0,7.5,10.0])
    plt.yticks([-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.25], [-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.25])
    
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    
    plt.xlim(-10, 10)
    plt.ylim(-0.10, 0.25)
    
    plt.xlabel("X", fontsize = 16)
    plt.ylabel("Probability", fontsize = 16)

    plt.grid()
    plt.show()

def evidence():
    fig = plt.figure(figsize = (5, 5))
    ax = fig.gca()
    ax.axis("off")

    ax.plot([0.2, 0.8], [0, 0], color = "black", linewidth = 10.0, zorder = 2)
    ax.plot([0.2, 0.8], [1, 1], color = "black", linewidth = 10.0, zorder = 2)
    ax.plot([0, 0], [0, 0], color = "orange", linewidth = 2.0, label = "L(Q)", zorder = 2)
    ax.plot([0, 0], [0, 0], color = "red", linewidth = 2.0, label = "KL(Q||P)", zorder = 2)
    ax.plot([0, 0], [0, 0], color = "magenta", linewidth = 2.0, label = "P(X)", zorder = 2)
    plt.arrow(0.4, 1.00, 0.00, -0.98, length_includes_head = True, facecolor = "orange", edgecolor = "orange", width = 0.01, head_width = 0.05, head_length = 0.05, zorder = 2)
    plt.arrow(0.6, 0.00, 0.00, 0.33, length_includes_head = True, facecolor = "red", edgecolor = "red", width = 0.01, head_width = 0.05, head_length = 0.05, zorder = 2)
    plt.arrow(0.6, 1.00, 0.00, -0.66, length_includes_head = True, facecolor = "magenta", edgecolor = "magenta", width = 0.01, head_width = 0.05, head_length = 0.05, zorder = 2)
    ax.legend(loc = "upper left", fontsize = 16, framealpha = 0)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()
    
def activation(activation):
    fig = plt.figure(figsize = (5, 5))
    ax = fig.gca()

    x = np.linspace(-10, 10, 1000)

    if (activation == "rectified"):
        y = np.maximum(x, 0)
        ax.plot(x, y, color = "orange", linewidth =  2.0, zorder = 2)
        
        plt.xticks([-10,-8,-6,-4,-2,0,2,4,6,8,10],[-10,-8,-6,-4,-2,0,2,4,6,8,10])
        plt.yticks([-10,-8,-6,-4,-2,0,2,4,6,8,10],[-10,-8,-6,-4,-2,0,2,4,6,8,10])
        
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:d}"))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:d}"))
        
        plt.xlim(-11,11)
        plt.ylim(-11,11)

    if (activation == "sigmoid"):
        y = 1 / (1 + np.exp(-x))
        ax.plot(x, y, color = "red", linewidth = 2.0, zorder = 2)
        
        plt.xticks([-10,-8,-6,-4,-2,0,2,4,6,8,10],[-10,-8,-6,-4,-2,0,2,4,6,8,10])
        plt.yticks([-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0],[-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0])
        
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:d}"))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))

        plt.xlim(-11,11)
        plt.ylim(-1.1,1.1)

    if (activation == "hyperbolic"):
        y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        ax.plot(x, y, color = "magenta", linewidth = 2.0, zorder = 2)
        
        plt.xticks([-10,-8,-6,-4,-2,0,2,4,6,8,10],[-10,-8,-6,-4,-2,0,2,4,6,8,10])
        plt.yticks([-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0],[-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0])
        
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:d}"))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))

        plt.xlim(-11,11)
        plt.ylim(-1.1,1.1)
    
    plt.xlabel("X", fontsize = 16)
    plt.ylabel("Activation", fontsize = 16)

    plt.grid()
    plt.show()

def optimisation():
    fig = plt.figure(figsize = (7, 5))
    ax = fig.gca(projection = "3d", proj_type = "ortho", azim = -135, elev = 25)
    col_map = continuous_cmap(["#FF00FF", "#FF00FF", "#FF0000","#FFAA00", "#FFAA00"], [0, 0.16, 0.33, 0.66, 1])

    x, y = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    z = 3 * np.square(1 - x) * np.exp(-np.square(x) - np.square(y + 1)) - 10 * (x / 5 - np.power(x, 3) - np.power(y, 5)) * np.exp(-np.square(x) - np.square(y)) - (1/3) * np.exp(-np.square(x + 1) - np.square(y))

    ax.plot_surface(x, y, z, edgecolor = "1", linewidth = 0.1, cmap = col_map)
    
    ax.set_xticks([-3,-2,-1,0,1,2,3])
    ax.set_yticks([-3,-2,-1,0,1,2,3])
    ax.set_zticks([-8,-4,0,4,8])

    ax.axes.set_xlim(left = -2.9, right = 2.9) 
    ax.axes.set_ylim(bottom = -2.9, top = 2.9) 
    ax.axes.set_zlim(bottom = -9.9, top = 9.9) 
    
    ax.set_xlabel("X", fontsize = 16)
    ax.set_ylabel("Y", fontsize = 16)
    ax.set_zlabel("Z", fontsize = 16)

    plt.tight_layout()
    plt.show()

def network(radius, layers, labels, colours):
    fig = plt.figure(figsize = (5, 5))
    ax = fig.gca()
    ax.axis("off")

    spacing = [1 / float(max(layers) - 1), 1 / float(len(layers) - 1)]

    positions = []
    for layer, nodes in enumerate(layers):
        position = []
        padding = (max(layers) - nodes) / 2
        for node in range(nodes):
            position.append([node * spacing[0] + padding * spacing[0], layer * spacing[1]])
        positions.append(position)

    for layer, nodes in enumerate(layers):
        for node in range(nodes):
            circle = plt.Circle((positions[layer][node][0], positions[layer][node][1]), radius = radius, facecolor = colours[layer], edgecolor = "black", linewidth = 2.0, zorder = 2)
            ax.add_patch(circle)
            string = f"{labels[layer]}" if (nodes == 1) else f"{labels[layer]}$\mathdefault{{_{str(node + 1)}}}$"
            plt.text(positions[layer][node][0], positions[layer][node][1], string, fontsize = 26, horizontalalignment = "center", verticalalignment = "center_baseline", zorder = 3)
    
    for layer, nodes in enumerate(layers[:-1]):
        for bot_node in range(nodes):
            for top_node in range(layers[layer + 1]):
                delta = [positions[layer + 1][top_node][0] - positions[layer][bot_node][0], positions[layer + 1][top_node][1] - positions[layer][bot_node][1]]
                scale = radius / np.sqrt(delta[0] ** 2 + delta[1] ** 2)
                plt.arrow(positions[layer][bot_node][0] + delta[0] * scale, positions[layer][bot_node][1] + delta[1] * scale, delta[0] - delta[0] * scale * 2, delta[1] - delta[1] * scale * 2, length_includes_head = True, facecolor = "black", edgecolor = "black", width = 0.0025, head_width = 0.03, head_length = 0.03, zorder = 2)

    plt.tight_layout()
    plt.show()

def results(paths):
    fig = plt.figure(figsize = (9, 3))
    ax = fig.gca()

    tasks = np.linspace(1, 10, 10) 
    labels = ["VCL - GIP (200)", "VCL - GIP (400)", "VCL - GIP (600)", "VCL - GIP (800)", "VCL - GIP (1000)", "VCL - GIP (1500)", "VCL - GIP (2000)"]
    colours = ["#FFAA00", "#FF7100", "#FF3900", "#FF0000", "#FF0055", "#FF00AA", "#FF00FF"]

    accuracies = []
    for path in paths:
        accuracies.append(np.loadtxt(path, delimiter=","))

    for count, _ in enumerate(accuracies):
        ax.plot(tasks, accuracies[::-1][count], color = colours[::-1][count], linewidth = 2.0, marker = "o", zorder = 2)
    for count, _ in enumerate(accuracies):
        ax.plot([0, 0], [0, 0], color = colours[count], linewidth = 2.0, marker = "o", label = labels[count], zorder = 2)
    ax.legend(loc = "upper left", bbox_to_anchor=(1, 1.05), fontsize = 12, framealpha = 0)

    plt.xticks([1,2,3,4,5,6,7,8,9,10], [-1,2,3,4,5,6,7,8,9,10])
    plt.yticks([0.90,0.92,0.94,0.96,0.98,1.00], [0.90,0.92,0.94,0.96,0.98,1.00])
    
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:d}"))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))

    plt.xlim(0.5, 10.5)
    plt.ylim(0.89, 1.01)

    plt.xlabel("Task", fontsize = 16)
    plt.ylabel("Accuracy", fontsize = 16)

    plt.tight_layout()
    plt.grid()
    plt.show()

def logs(path, log):
    fig = plt.figure(figsize = (7, 5))
    ax = fig.gca()

    if (log == "evidence"):
        _, step, value = np.loadtxt((path[0] / "evidence.csv"), delimiter = ",", unpack = True)

        ax.plot(step, -value, color = "orange", linewidth =  2.0, zorder = 2)
        
        plt.xticks([0,100000,200000,300000,400000,500000],[0,100000,200000,300000,400000,500000])
        plt.yticks([-0.40,-0.35,-0.30,-0.25,-0.20,-0.15],[-0.40,-0.35,-0.30,-0.25,-0.20,-0.15])
        
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:d}"))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        
        plt.xlim(0,500000)
        plt.ylim(-0.40,-0.15)

    if (log == "loss"):
        _, step, value = np.loadtxt((path[0] / "loss.csv"), delimiter = ",", unpack = True)

        ax.plot(step, value, color = "red", linewidth =  2.0, zorder = 2)
        
        plt.xticks([0,100000,200000,300000,400000,500000],[0,100000,200000,300000,400000,500000])
        plt.yticks([0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26],[0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26])
        
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:d}"))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        
        plt.xlim(0,500000)
        plt.ylim(0.10,0.26)

    if (log == "accuracy"):
        _, step, value = np.loadtxt((path[0] / "accuracy.csv"), delimiter = ",", unpack = True)

        ax.plot(step, value, color = "magenta", linewidth =  2.0, zorder = 2)
        
        plt.xticks([0,100000,200000,300000,400000,500000],[0,100000,200000,300000,400000,500000])
        plt.yticks([0.92,0.93,0.94,0.95,0.96,0.97,0.98],[0.92,0.93,0.94,0.95,0.96,0.97,0.98])
        
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:d}"))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        
        plt.xlim(0,500000)
        plt.ylim(0.92,0.98)


    plt.xlabel("Step", fontsize = 16)
    plt.ylabel("Value", fontsize = 16)

    plt.grid()
    plt.show()

if __name__ == "__main__":
    main(parser.parse_args())