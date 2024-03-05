import argparse
import logging

from src.models.train import Trainer
from src.models.eval import Evaluator
from src.uncertainty.analyze import Analyzer


def train(args):
    try:
        trainer = Trainer(args)
        trainer.train()
        trainer.save_model()
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")


def evaluate(args):
    try:
        evaluator = Evaluator(args)
        evaluator.evaluate()
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")


def analyze(args):
    try:
        analyzer = Analyzer(args)
        analyzer.analyze()
    except Exception as e:
        logging.error(f"An error occurred during uncertainty analysis: {e}")


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="AI Model Uncertainty Analysis Tool")
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand for training
    parser_train = subparsers.add_parser('train', help='Train the model')
    parser_train.add_argument('--epochs', type=int, default=5, help='Number of epochs for training the model')
    parser_train.add_argument('--batch', type=int, default=64, help='Type in batch size for training')
    parser_train.add_argument('--adv', action='store_true', help='Enable adversarial training')
    parser_train.add_argument('--eps', type=float, default=0.3,
                              help='Epsilon value for adversarial training, only used if --adv-train is specified')
    parser_train.add_argument('--save-path', type=str, default=None, help='Path to save the model weights')
    parser_train.set_defaults(func=train)

    # Subcommand for evaluation
    parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate the model')
    parser_evaluate.add_argument('--adv-eval', action='store_true', help='Perform adversarial evaluation')
    parser_evaluate.add_argument('--model-path', type=str, default=None, help='Path to save the model weights')
    parser_evaluate.add_argument('--eps', type=float, default=0.3, help='Epsilon for adversarial perturbation')
    parser_evaluate.set_defaults(func=evaluate)

    # Subcommand for uncertainty analysis
    uncertainty_parser = subparsers.add_parser('analyze', help='Analyze model uncertainty')
    uncertainty_parser.add_argument('--model-path', type=str, default=None, help='Path to load the model weights')
    uncertainty_parser.add_argument('--batch', type=int, default=64, help='Type in batch size for analyzing')
    uncertainty_parser.set_defaults(func=analyze)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except Exception as e:
            logging.error(f"An error occurred when executing the {args.command} command: {e}")
            parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
