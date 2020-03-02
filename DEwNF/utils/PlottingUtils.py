import matplotlib.pyplot as plt


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
