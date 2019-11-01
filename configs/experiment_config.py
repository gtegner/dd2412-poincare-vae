import pandas as pd
from uuid import uuid4


def update_config(cfg, default):
    a = default.copy()
    a.update(cfg)
    return a


default_params = {
    "model": 'hyperbolic',
    "dataset": 'toy',
    "epochs": 1000,
    "batch_size": 64,
    "test_batch": 64,
    "distribution": 'riemannian',
    "lr": 1e-3,
    "save_model": 1,
    "log_interval": 1,
    "latent_dim": 2,
    "beta": 1,
    "checkpoint_interval": 1000 // 10,
    "toy_dim": 50,
    "prior_sigma": 1.0,
    "c": 1.0,
    "seed": 42,
    "experiment_name": "toy",
    "break_early": 0,
    "break_interval": 1000,
    "loss": "mse",
    "hidden_dim": 200,
    "marginal_samples": 300,
    "model_uuid": "1234",
    "test_interval": 10,

    "mu_loss": 0,
    "recon_loss": 0,
    "kl_loss": 0,
    "mode": "train/test",
    "epoch": 0,
}

toy_test = {
    "model": 'hyperbolic',
    "dataset": 'toy',
    "epochs": 100,
    "distribution": 'riemannian',
    "log_interval": 10,
    "latent_dim": 2,
    "checkpoint_interval": 100,
    "test_interval": 1,
    "marginal_samples": 1000,
    "toy_dim": 50,
    "experiment_name": "toy-test",
    "hidden_dim": 200,
}


toy = {
    "model": 'hyperbolic',
    "dataset": 'toy',
    "epochs": 1000,
    "batch_size": 64,
    "test_batch": 64,
    "distribution": 'riemannian',
    "lr": 1e-3,
    "log_interval": 10,
    "latent_dim": 2,
    "beta": 1,
    "checkpoint_interval": 100,
    "toy_dim": 50,
    "prior_sigma": 1.0,
    "c": 1.0,
    "seed": 42,
    "experiment_name": "toy",
    "loss": "mse",
    "hidden_dim": 200,
    "test_interval": 1000,
    "marginal_samples": 1000
}


mnist = {
    "model": 'hyperbolic',
    "dataset": 'mnist',
    "epochs": 500,
    "batch_size": 128,
    "test_batch": 128,
    "distribution": 'normal',
    "log_interval": 1,
    "latent_dim": 2,
    "beta": 1,
    "checkpoint_interval": 10,
    "prior_sigma": 1.0,
    "c": 1.0,
    "seed": 42,
    "experiment_name": "mnist",
    "break_early": 0,
    "break_interval": 30,
    "loss": "bce",
    "hidden_dim": 600,
    "test_interval": 500,
    "marginal_samples": 500
}

test_mnist = {
    "model": 'hyperbolic',
    "dataset": 'mnist',
    "epochs": 1,
    "batch_size": 128,
    "test_batch": 128,
    "distribution": 'riemannian',
    "log_interval": 1,
    "latent_dim": 5,
    "beta": 1,
    "checkpoint_interval": 10,
    "prior_sigma": 1.0,
    "c": 1.0,
    "seed": 42,
    "experiment_name": "mnist",
    "break_early": 1,
    "break_interval": 5,
    "loss": "bce",
    "hidden_dim": 32,
    "test_interval": 100,
    "marginal_samples": 5
}

toy_test = update_config(toy_test, default_params)
toy = update_config(toy, default_params)
mnist = update_config(mnist, default_params)
test_mnist = update_config(test_mnist, default_params)


def toy_test_exp(experiment_name):
    default_params = toy_test
    experiments = [default_params]

    df = pd.DataFrame(columns=experiments[0].keys())
    return df, experiments


def mnist_test(experiment_name):
    default_params = test_mnist
    experiments = [default_params]

    df = pd.DataFrame(columns=experiments[0].keys())
    return df, experiments


def toy_full(experiment_name):
    default_params = toy
    default_params['experiment_name'] = experiment_name

    cs = list(reversed([0.1, 0.3, 0.8, 1.0]))
    prior_sigma = list(reversed([1.0, 1.3, 1.7]))

    experiments = []

    dist = ['riemannian']
    # Riemannian Normal
    for d in dist:
        for c in cs:
            for prior in prior_sigma:
                params = {
                    'c': c,
                    'prior_sigma': prior,
                    'model': 'hyperbolic',
                    'distribution': d,
                    'experiment_name': 'toy_riemannian_importance',
                    'model_uuid': str(uuid4())[0:5]
                }
                new_params = default_params.copy()
                new_params.update(params)
                experiments.append(new_params)

    for prior in prior_sigma:
        params = {
            'prior_sigma': prior,
            'model': 'vanilla',
            'distribution': 'normal',
            'experiment_name': 'toy_vanilla_importance',
            'model_uuid': str(uuid4())[0:5]
        }
        new_params = default_params.copy()
        new_params.update(params)
        experiments.append(new_params)

    df = pd.DataFrame(columns=experiments[0].keys())

    return df, experiments



def mnist_experiment(experiment_name):
    default_params = mnist
    default_params['experiment_name'] = experiment_name

    cs = [0.1, 0.2, 0.7, 1.4]
    # cs = [1.4]
    prior_sigma = 1.0
    distributions = ['riemannian', 'wrapped']
    latent_dims = [2, 5, 10, 20]
    loss = ['bce']
    experiments = []

    # Riemannian Normal
    for ls in loss:
        for dim in latent_dims:
            for dist in distributions:
                for c in cs:
                    params = {
                        'c': c,
                        'prior_sigma': prior_sigma,
                        'model': 'hyperbolic',
                        'distribution': dist,
                        'experiment_name': experiment_name,
                        'model_uuid': str(uuid4())[0:5],
                        'latent_dim': dim,
                        "loss": ls
                    }
                    new_params = default_params.copy()
                    new_params.update(params)
                    experiments.append(new_params)

    for dim in latent_dims:
        params = {
            'prior_sigma': prior_sigma,
            'model': 'vanilla',
            'distribution': 'normal',
            'experiment_name': experiment_name,
            'model_uuid': str(uuid4())[0:5],
            'latent_dim': dim,
            "loss": 'bce'
        }
        new_params = default_params.copy()
        new_params.update(params)
        experiments.append(new_params)

    df = pd.DataFrame(columns=experiments[0].keys())
    return df, experiments


