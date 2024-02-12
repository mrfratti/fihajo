import argparse
from training import train_model
from evaluation import evaluate_model
from uncertainty_analysis import analyze_uncertainty


def main():
    parser = argparse.ArgumentParser(description="AI Model Uncertainty Analysis Tool")
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand for training
    parser_train = subparsers.add_parser('train', help='Train the model')
    parser_train.add_argument('--epochs', type=int, default=5, help='Number of epochs for training the model')
    parser_train.add_argument('--batch', type=int, default=64, help='Type in batch size for training')
    parser_train.set_defaults(func=train_model)

    # Subcommand for evaluation
    parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate the model')
    parser_evaluate.add_argument('--eps', type=float, default=0.3, help='Epsilon for adversarial perturbation')
    parser_evaluate.set_defaults(func=evaluate_model)

    # Subcommand for uncertainty analysis
    uncertainty_parser = subparsers.add_parser('analyze', help='Analyze model uncertainty')
    uncertainty_parser.add_argument('--batch', type=int, default=64, help='Type in batch size for analyzing')
    uncertainty_parser.set_defaults(func=analyze_uncertainty)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()