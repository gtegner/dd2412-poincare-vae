import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Which dataset", default="toy")
parser.add_argument("--epochs", help="Number of epochs", default=10, type=int)
parser.add_argument("--batch_size", help="Batch size", default=64, type=int)
parser.add_argument(
    "--test_batch", help="Size of test batch", default=64, type=int)
parser.add_argument("--c", help="Curvature", default=1.2, type=float)
parser.add_argument(
    "--distribution", help="Wrapped normal or riemannian", default="riemannian")
parser.add_argument("--lr", help="Learning rate", default=1e-3, type=float)
parser.add_argument("--seed", help="Torch random seed", default=42, type=int)
parser.add_argument("--save_model", help="Save model", default=1, type=int)
parser.add_argument("--log_interval", help="Print interval",
                    default=10, type=int)
parser.add_argument("--latent_dim", help="Latent dimension",
                    default=2, type=int)
parser.add_argument("--beta", help="Influence of kl-div",
                    default=1, type=float)
parser.add_argument("--checkpoint-interval", help="Save models every",
                    default=1, type=int)
parser.add_argument("--toy_dim", help="Dimension of toy dataset",
                    default=50, type=int, required=False)
parser.add_argument("--prior_sigma", help="Prior variance",
                    default=1.0, type=float, required=False)
parser.add_argument("--model_uuid", help="Unique identifier",
                    default="test-123", type=str, required=False)
parser.add_argument("--model", help="Model",
                    default="hyperbolic", type=str, required=False)

config, _ = parser.parse_known_args()
