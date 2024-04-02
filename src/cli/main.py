import argparse
import logging
import json
import tensorflow as tf
from uncertainty_wizard.models import StochasticMode
from src.cli.send.send_report_data import SendReportData
from src.datasets.dataset_handler import (
    MnistDatasetHandler,
    Cifar10DatasetHandler,
    FashionMnistDatasetHandler,
)
from src.models.model_builders import (
    MNISTModelBuilder,
    Cifar10ModelBuilder,
    FashionMnistModelBuilder,
)
from src.models.train import Trainer
from src.models.eval import Evaluator
from src.uncertainty.analyze import Analyzer


class CLIApp:
    def __init__(self):
        # tf.config.set_visible_devices([], "GPU")  # uncomment to disable gpu
        self.parser = self.setup_parser()
        self._plot_file_names = {}

    def setup_parser(self):
        parser = argparse.ArgumentParser(
            description="AI Model Uncertainty Analysis Tool"
        )
        parser.add_argument(
            "--config",
            type=str,
            help="Path to a JSON configuration file. Command line arguments "
            "override config file values.",
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Increase output verbosity."
        )
        parser.add_argument(
            "--quiet",
            action="store_true",
            help="Minimize the output, show only essential information.",
        )

        subparsers = parser.add_subparsers(dest="command")

        self.add_train_subparser(subparsers)
        self.add_evaluate_subparser(subparsers)
        self.add_analyze_subparser(subparsers)
        self.add_report_subparser(subparsers)

        return parser

    def add_train_subparser(self, subparsers):
        parser_train = subparsers.add_parser("train", help="Train the model")
        parser_train.add_argument(
            "--dataset",
            type=str,
            choices=["mnist", "cifar10", "fashion_mnist"],
            required=True,
            help="The dataset to use for training.",
        )
        parser_train.add_argument(
            "--epochs",
            type=self.check_positive,
            default=5,
            help="Number of epochs for " "training the model",
        )
        parser_train.add_argument(
            "--batch",
            type=self.check_positive,
            default=64,
            help="Batch size for training",
        )
        parser_train.add_argument(
            "--adv",
            action="store_true",
            help="Enable adversarial training to improve model "
            "robustness against adversarial examples.",
        )
        parser_train.add_argument(
            "--eps",
            type=self.check_eps,
            default=0.3,
            help="Epsilon value for adversarial training, controlling the perturbation magnitude.",
        )
        parser_train.add_argument(
            "--save-path", type=str, default=None, help="Path to save the model weights"
        )
        parser_train.add_argument(
            "--optimizer",
            type=str,
            default="adadelta",
            choices=["adadelta", "adam", "sgd"],
            help="Optimizer for training",
        )
        parser_train.add_argument(
            "--learning-rate",
            type=float,
            default=None,
            help="Learning rate for the optimizer",
        )
        parser_train.add_argument(
            "--report", action="store_true", help="Generate report"
        )

    def add_evaluate_subparser(self, subparsers):
        parser_evaluate = subparsers.add_parser("evaluate", help="Evaluate the model")
        parser_evaluate.add_argument(
            "--model-path",
            type=str,
            default=None,
            help="Path to the the model weights for " "evaluation",
        )
        parser_evaluate.add_argument(
            "--dataset",
            type=str,
            choices=["mnist", "cifar10", "fashion_mnist"],
            required=True,
            help="The dataset used for training.",
        )
        parser_evaluate.add_argument(
            "--adv-eval",
            action="store_true",
            help="Perform adversarial evaluation to test " "model robustness.",
        )
        parser_evaluate.add_argument(
            "--eps",
            type=self.check_eps,
            default=0.3,
            help="Epsilon for adversarial " "perturbation during evaluation",
        )
        parser_evaluate.add_argument(
            "--report", action="store_true", help="Generate report"
        )
        parser_evaluate.set_defaults(func=self.evaluate)

    def add_analyze_subparser(self, subparsers):
        uncertainty_parser = subparsers.add_parser(
            "analyze", help="Analyze model uncertainty"
        )
        uncertainty_parser.add_argument(
            "--dataset",
            type=str,
            choices=["mnist", "cifar10", "fashion_mnist"],
            required=True,
            help="The dataset used for analysis.",
        )
        uncertainty_parser.add_argument(
            "--model-path",
            type=str,
            default=None,
            help="Path to load the model weights " "for uncertainty analysis.",
        )
        uncertainty_parser.add_argument(
            "--batch", type=int, default=64, help="Batch size for analyzing."
        )
        uncertainty_parser.set_defaults(func=self.analyze)

    def add_report_subparser(self, subparsers):
        report_parser = subparsers.add_parser("report", help="Genereate report")
        report_parser.set_defaults(func=self.report)

    def check_positive(self, value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                f"{value} is an invalid positive int value"
            )
        return ivalue

    def check_eps(self, value):
        fvalue = float(value)
        if fvalue <= 0.0 or fvalue > 1.0:
            raise argparse.ArgumentTypeError(
                f"{value} is out of the allowed range (0.0, 1.0)"
            )
        return fvalue

    def train(self, args):
        stochastic_mode = StochasticMode()
        # Instantiate the correct model builder based on the command line argument
        dataset_handlers = {
            "mnist": MnistDatasetHandler(),
            "cifar10": Cifar10DatasetHandler(),
            "fashion_mnist": FashionMnistDatasetHandler(),
        }

        dataset_handler = dataset_handlers[args.dataset]

        # Instantiate the correct model builder based on the command line argument
        model_builders = {
            "mnist": MNISTModelBuilder,
            "cifar10": Cifar10ModelBuilder,
            "fashion_mnist": FashionMnistModelBuilder,
        }

        model_builder = model_builders[args.dataset](
            stochastic_mode, args.optimizer, args.learning_rate
        )

        (x_train, y_train), (x_test, y_test) = dataset_handler.load_and_preprocess()

        try:
            trainer = Trainer(model_builder, (x_train, y_train), (x_test, y_test), args)
            trainer.train()
            trainer.save_model()
            self._plot_file_names.update(trainer.plot_file_names)
            if args.report:
                self.report()
        except Exception as e:
            logging.error("An error occurred during training: %s", e)

    def evaluate(self, args):
        """Runs evaluation attack on model"""
        model_path = (
            args.model_path
            if hasattr(args, "model_path") and args.model_path
            else input(
                "Enter the model path for analysis or press Enter to use the default path: "
            ).strip()
        )
        if not model_path:
            model_path = Evaluator.default_path

        # logging.info(f"Evaluating model from {model_path} on {args.dataset} dataset")

        if not hasattr(args, "adv_eval"):
            args.adv_eval = False

        stochastic_mode = StochasticMode()
        # Instantiate the correct model builder based on the command line argument
        dataset_handlers = {
            "mnist": MnistDatasetHandler(),
            "cifar10": Cifar10DatasetHandler(),
            "fashion_mnist": FashionMnistDatasetHandler(),
        }

        dataset_handler = dataset_handlers[args.dataset]

        # Instantiate the correct model builder based on the command line argument
        model_builders = {
            "mnist": MNISTModelBuilder(stochastic_mode),
            "cifar10": Cifar10ModelBuilder(stochastic_mode),
            "fashion_mnist": FashionMnistModelBuilder(stochastic_mode),
        }

        model_builder = model_builders[args.dataset]

        (x_test, y_test) = dataset_handler.load_and_preprocess()

        try:
            evaluator = Evaluator(model_builder, (x_test, y_test), args)
            evaluator.evaluate()
            self._plot_file_names.update(evaluator.plot_file_names)
            if args.report:
                self.report()
        except Exception as e:
            logging.error("An error occurred during evaluation: %s", e)

    def analyze(self, args):
        model_path = (
            args.model_path
            if hasattr(args, "model_path") and args.model_path
            else input(
                "Enter the model path for analysis or press Enter to use the default path: "
            ).strip()
        )
        if not model_path:
            model_path = Analyzer.default_path

        batch = getattr(args, "batch", 64)

        # Instantiate the correct model builder based on the command line argument
        dataset_handlers = {
            "mnist": MnistDatasetHandler(),
            "cifar10": Cifar10DatasetHandler(),
            "fashion_mnist": FashionMnistDatasetHandler(),
        }

        # Instantiate the correct model builder based on the command line argument
        model_builders = {
            "mnist": MNISTModelBuilder(StochasticMode()),
            "cifar10": Cifar10ModelBuilder(StochasticMode()),
            "fashion_mnist": FashionMnistModelBuilder(StochasticMode()),
        }

        model_builder = model_builders[args.dataset]
        dataset_handler = dataset_handlers[args.dataset]

        (x_test, y_test) = dataset_handler.load_and_preprocess()

        try:
            analyzer = Analyzer(model_builder, (x_test, y_test), batch, args)
            analyzer.analyze()
        except Exception as e:
            logging.error("An error occurred during uncertainty analysis: %s", e)

    def report(self):
        """Run report generation"""
        try:
            send = SendReportData()
            send.filenames = self._plot_file_names
            send.send()
        except ValueError as e:
            logging.warning("main.report: %s", e)

        except TypeError as e:
            logging.warning("main.report: %s", e)

    def load_config(self, file_path):
        """loading predefined configuration file in json format"""
        with open(file_path, "r", encoding="UTF-8") as f:
            return json.load(f)

    def run(self):
        self.args = self.parser.parse_args()
        if self.args.config:
            config_args = self.load_config(self.args.config)
            self.args.__dict__.update(config_args)
            # config_args.setdefault('adv', False)
            # for key, value in config_args.items():
            #    setattr(args, key, value)

        if self.args.verbose:
            logging.basicConfig(
                level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
            )
        elif self.args.quiet:
            logging.basicConfig(
                level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
            )
        else:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
            )

        if hasattr(self.args, "command") and self.args.command:
            getattr(self, self.args.command)(self.args)
        else:
            self.parser.print_help()
            exit(1)


if __name__ == "__main__":
    app = CLIApp()
    app.run()
