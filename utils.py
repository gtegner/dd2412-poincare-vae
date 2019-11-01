import torch
import matplotlib.pyplot as plt
import numpy as np
import os

ROOT_DIR = 'model_outputs'

CSV_DIR = os.path.join(ROOT_DIR, 'saved_csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
IMAGES_DIR = os.path.join(ROOT_DIR, 'saved_images')


def set_up_figure(model_type, c=1):
    if model_type in ['hyperbolic']:
        radius = 1/np.sqrt(c)
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.cla()
        ax.set_xlim((-radius*1.1, radius*1.1))
        ax.set_ylim((-radius*1.1, radius*1.1))
        ax.add_artist(plt.Circle((0, 0), radius, color='black', fill=False))
    elif model_type in ['vanilla']:
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.cla()

    return fig, ax


def plot_z(z, c, label, model_type='hyperbolic', title='', dataset='toy'):
    if isinstance(z, torch.Tensor):
        z = z.detach().numpy()

    fig, ax = set_up_figure(model_type, c)

    if label is not None:
        if isinstance(label, torch.Tensor):
            label = np.array(np.int_(label.detach().numpy()))
        scatter = ax.scatter(z[:, 0], z[:, 1], marker='.',
                             c=label.ravel().tolist())

        post_means = get_post_means(z, label)
        scatter_means = ax.scatter(
            post_means[:, 0], post_means[:, 1], c='k', marker='o')

        if dataset in ['toy']:
            graph = get_graph(label)
            for key, item in graph.items():
                mu_key = post_means[np.where(post_means[:, 2] == key)][0, :2]
                for child in item:
                    chi = post_means[np.where(
                        post_means[:, 2] == child)][0, :2]
                    ax.plot([mu_key[0], chi[0]], [mu_key[1], chi[1]],
                            color='k', linestyle=':')
        elif dataset in ['mnist']:
            ax.legend(*scatter.legend_elements(),
                      loc='best', bbox_to_anchor=(1, 1))
            # pass

        plt.suptitle(title)
        return fig

    else:
        ax.plot(z[:, 0], z[:, 1], 'o', color='y')


def get_post_means(z, label):
    _, dim = z.shape
    if isinstance(z, torch.Tensor):
        z = z.detach().numpy()
    if isinstance(label, torch.Tensor):
        label = label.detach().numpy()

    post_means = []
    for number in np.unique(label):
        mu_number = np.where(label == number)
        x_num = z[mu_number]
        mean_num = np.mean(x_num[:, 0:dim], 0)
        mean_flattened = [mean_num[i] for i in range(len(mean_num))]
        mean_flattened.append(number)
        post_means.append(mean_flattened)

    post_means = np.asarray(post_means)
    return post_means


def get_graph(labels):
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().numpy()

    graph = {}
    m = np.max(np.unique(labels))
    for label in np.unique(labels):
        child1 = 2 * label + 1
        child2 = 2 * label + 2
        if child1 <= m and child2 <= m:
            graph[label] = [child1, child2]
        else:
            graph[label] = []

    return graph


def plot_and_save_fig(model, c, data_loader, file_name, typ='hyperbolic', args=None):

    mu_total = []
    labels = []
    with torch.no_grad():
        for ix, (xsample, label) in enumerate(data_loader):
            if args.dataset in ['mnist']:
                xsample = xsample.view(xsample.size(0), -1)

            if typ in ['hyperbolic']:
                decoded, mu, sigma, log_prob, prior_log_prob = model(xsample)
            else:
                decoded, mu, sigma = model(xsample)

            mu_total.append(mu)
            labels.append(label)

    mu_total = torch.cat(mu_total, 0)
    labels_total = torch.cat(labels).view(-1)

    try:
        fig = plot_z(mu_total, c=c, label=labels_total,
                     model_type=typ, title=f'{typ} VAE latent space', dataset=args.dataset, return_ax=return_ax)

        fig.savefig(file_name)
        plt.close(fig)
    except Exception as e:
        print("Error saving figure")
        print(args)
        print(e)
    return 0


def write_row_to_df(args, df):
    if df is None:
        print("Cant write row")
        return None
    kwargs = vars(args)
    df = df.append(kwargs, ignore_index=True)
    return df


def safe_log(x, eps=1e-8):
    return torch.log(x + eps)
