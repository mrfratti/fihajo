import argparse
import logging
import json

from src.models.train import Trainer
from src.models.eval import Evaluator
from src.uncertainty.analyze import Analyzer


class CLIApp:
    def __init__(self):
        self.parser = self.setup_parser()

    def setup_parser(self):
        parser = argparse.ArgumentParser(description="AI Model Uncertainty Analysis Tool")
        parser.add_argument('--config', type=str, help='Path to a JSON configuration file. Command line arguments '
                                                       'override config file values.')
        parser.add_argument('--verbose', action='store_true', help='Increase output verbosity.')
        parser.add_argument('--quiet', action='store_true',
                            help='Minimize the output, show only essential information.')

        subparsers = parser.add_subparsers(dest="command")

        self.add_train_subparser(subparsers)
        self.add_evaluate_subparser(subparsers)
        self.add_analyze_subparser(subparsers)

        return parser

    def add_train_subparser(self, subparsers):
        parser_train = subparsers.add_parser('train', help='Train the model')
        parser_train.add_argument('--epochs', type=self.check_positive, default=5, help='Number of epochs for '
                                                                                        'training the model')
        parser_train.add_argument('--batch', type=self.check_positive, default=64, help='Batch size for training')
        parser_train.add_argument('--adv', action='store_true', help='Enable adversarial training to improve model '
                                                                     'robustness against adversarial examples.')
        parser_train.add_argument('--eps', type=self.check_eps, default=0.3,
                                  help='Epsilon value for adversarial training, controlling the perturbation magnitude.')
        parser_train.add_argument('--save-path', type=str, default=None, help='Path to save the model weights')
        #parser_train.add_argument('--optimizer', type=str, default='adadelta', help='Optimizer for training')
        #parser_train.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for the optimizer')
        #parser_train.add_argument('--checkpoint-path', type=str, default=None,
        #                          help='Path to save checkpoints during training')
        #parser_train.add_argument('--validation-split', type=float, default=0.1,
        #                          help='Fraction of the training data to be used as validation data')

        #parser_train.set_defaults(func=self.train)

    def add_evaluate_subparser(self, subparsers):
        parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate the model')
        parser_evaluate.add_argument('--model-path', type=str, default=None, help='Path to the the model weights for '
                                                                                  'evaluation')
        parser_evaluate.add_argument('--adv-eval', action='store_true', help='Perform adversarial evaluation to test '
                                                                             'model robustness.')
        parser_evaluate.add_argument('--eps', type=self.check_eps, default=0.3, help='Epsilon for adversarial '
                                                                                     'perturbation during evaluation')
        parser_evaluate.set_defaults(func=self.evaluate)

    def add_analyze_subparser(self, subparsers):
        uncertainty_parser = subparsers.add_parser('analyze', help='Analyze model uncertainty')
        uncertainty_parser.add_argument('--model-path', type=str, default=None, help='Path to load the model weights '
                                                                                     'for uncertainty analysis.')
        uncertainty_parser.add_argument('--batch', type=int, default=64, help='Batch size for analyzing.')
        uncertainty_parser.set_defaults(func=self.analyze)

    def check_positive(self, value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
        return ivalue

    def check_eps(self, value):
        fvalue = float(value)
        if fvalue <= 0.0 or fvalue > 1.0:
            raise argparse.ArgumentTypeError(f"{value} is out of the allowed range (0.0, 1.0)")
        return fvalue

    def train(self, args):
        try:
            trainer = Trainer(args)
            trainer.train()
            trainer.save_model()
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")

    def evaluate(self, args):
        if not hasattr(args, 'model_path') or not args.model_path:
            args.model_path = input("Enter the model path for evaluation of press Enter to use the default path: ").strip()
            if not args.model_path:
                args.model_path = Evaluator._default_load_path()

        if not hasattr(args, 'adv_eval'):
            args.adv_eval = False

        try:
            evaluator = Evaluator(args)
            evaluator.evaluate()
        except Exception as e:
            logging.error(f"An error occurred during evaluation: {e}")

    def analyze(self, args):
        model_path = args.model_path if hasattr(args, 'model_path') and args.model_path else input(
            "Enter the model path for analysis or press Enter to use the default path: ").strip()
        if not model_path:
            model_path = Analyzer._default_load_path()
        batch = getattr(args, 'batch', 64)

        try:
            analyzer = Analyzer(model_path=model_path, batch=batch)
            analyzer.analyze()
        except Exception as e:
            logging.error(f"An error occurred during uncertainty analysis: {e}")

    def load_config(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def run(self):
        self.args = self.parser.parse_args()
        if self.args.config:
            config_args = self.load_config(self.args.config)
            self.args.__dict__.update(config_args)
            #config_args.setdefault('adv', False)
            #for key, value in config_args.items():
            #    setattr(args, key, value)

        if self.args.verbose:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        elif self.args.quiet:
            logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        #if hasattr(args, 'func'):
        #    try:
        #        args.func(args)
        #    except Exception as e:
        #        logging.error(f"An error occurred when executing the {args.command} command: {e}")
        #        self.parser.print_help()
        #else:
        #    self.parser.print_help()
        # ---------------------------------------
        #if hasattr(args, 'command'):
        #    if args.command == 'train':
        #        self.train(args)
        #    elif args.command == 'evaluate':
        #        self.evaluate(args)
        #    elif args.command == 'analyze':
        #        self.analyze(args)
        #    else:
        #        self.parser.print_help()
        #else:
        #    self.parser.print_help()
        if hasattr(self.args, 'command'):
            getattr(self, self.args.command)(self.args)
        else:
            self.parser.print_help()


if __name__ == "__main__":
    app = CLIApp()
    app.run()
