import matplotlib.pyplot as plt
import numpy as np


def plot_4_contexts_cond_flow(flow_dist, contexts, n_samples=256):
    assert contexts.shape[0] == 4, 'Need 4 contexts inorder to create 4 plots'

    fig, axs = plt.subplots(2, 2)
    for i in range(4):
        cur_axs = axs[i // 2, i % 2]
        cond_dist = flow_dist.condition(contexts[i])
        x_s = cond_dist.sample((n_samples,))
        cur_axs.scatter(x_s[:, 0].cpu(), x_s[:, 1].cpu(), c='b', s=5)
        cur_axs.set_xlim(-6, 6)
        cur_axs.set_ylim(-6, 6)
        context_str = np.array2string(contexts[i].cpu().numpy(), precision=2, separator=',')
        cur_axs.set_title(f"Context: {context_str}")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()


def plot_loss(train_loss_arr, test_loss_arr):
    """
    Makes nice plots of the train and test loss
    :param train_loss_arr: array
    :param test_loss_arr: array
    :return:
    """
    plt.figure()
    plt.title("Losses")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_loss_arr)
    plt.plot(test_loss_arr, '--')
    plt.legend(['Train', 'Test'])
    plt.show()


def sliding_plot_loss(train_loss_arr, test_loss_arr, window_size):
    """
    Makes nice plots of the train and test loss
    :param train_loss_arr: array
    :param test_loss_arr: array
    :return:
    """
    plt.figure()
    plt.title("Losses")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_loss_arr[-window_size:])
    plt.plot(test_loss_arr[-window_size:], '--')
    plt.legend(['Train', 'Test'])
    plt.show()


def plot_samples(plot_flow_dist, x_plot, n_samples=256):
    x_s = plot_flow_dist.sample((n_samples,))
    plt.figure(figsize=(4, 4))
    plt.scatter(x_plot[:, 0], x_plot[:, 1], c='r', s=5)
    plt.scatter(x_s[:, 0], x_s[:, 1], c='b', s=5)
    plt.axis('equal')
    plt.show()