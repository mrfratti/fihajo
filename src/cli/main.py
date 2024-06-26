import argparse
import logging
import json

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
    """Command-line interface for AI Model Uncertainty Analysis"""

    def __init__(self):
        """Initializes CLIApp with argument parser setup."""
        self.parser = self.setup_parser()
        self._report_location = "data/send.json"
        self._plot_file_names = {}
        self._interactive_report_location = "data/send_interactive.json"
        self._interactive_plot_file_names = {}

    def setup_parser(self):
        parser = argparse.ArgumentParser(
            description="AI Model Uncertainty Analysis Tool"
        )
        parser.add_argument(
            "--config",
            type=str,
            help="Path to a JSON configuration file. Command line arguments override"
            " config file values.",
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
            help="Epsilon value for adversarial "
            "training, controlling the "
            "perturbation magnitude.",
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
        parser_train.add_argument(
            "--interactive",
            action="store_true",
            help="Generate interactive plot for report",
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
        parser_evaluate.add_argument(
            "--interactive",
            action="store_true",
            help="Generate interactive plot for report",
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
        uncertainty_parser.add_argument(
            "--report", action="store_true", help="Generate report"
        )
        uncertainty_parser.add_argument(
            "--interactive",
            action="store_true",
            help="Generate interactive plot for report",
        )
        uncertainty_parser.set_defaults(func=self.analyze)

    def add_report_subparser(self, subparsers):
        report_parser = subparsers.add_parser("report", help="Generate report")
        report_parser.add_argument(
            "--interactive", action="store_true", help="Generate interactive report"
        )
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
        # Instantiate the correct model builder based on the command line
        # argument
        dataset_handlers = {
            "mnist": MnistDatasetHandler(),
            "cifar10": Cifar10DatasetHandler(),
            "fashion_mnist": FashionMnistDatasetHandler(),
        }

        dataset_handler = dataset_handlers[args.dataset]

        # Instantiate the correct model builder based on the command line
        # argument
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
            # process
            trainer = Trainer(model_builder, (x_train, y_train), (x_test, y_test), args)
            trainer.train()
            trainer.save_model()

            # find and save to data
            SendReportData(self._report_location).filenames = trainer.plot_file_names
            if args.report:
                self.report()

            if args.interactive:
                SendReportData(self._interactive_report_location).filenames = (
                    trainer.interactive_plot_file_names
                )
                # self.reportInteractive()

        except Exception as e:
            logging.error("An error occurred during training: %s", e)

    def evaluate(self, args):
        """Runs evaluation attack on model"""

        if hasattr(args, "model_path") and args.model_path is not None:
            model_path = args.model_path
        else:
            model_path = input(
                "Enter the model path for analysis or press Enter to use the default path: "
            ).strip()

        if not model_path:
            logging.info("No path set defaulting to defualt path \n")
            model_path = Evaluator.default_path

        logging.info(
            "Evaluating model from %s on %s dataset \n", model_path, args.dataset
        )

        if not hasattr(args, "adv_eval"):
            args.adv_eval = False

        stochastic_mode = StochasticMode()
        # Instantiate the correct model builder based on the command line
        # argument
        dataset_handlers = {
            "mnist": MnistDatasetHandler(),
            "cifar10": Cifar10DatasetHandler(),
            "fashion_mnist": FashionMnistDatasetHandler(),
        }

        dataset_handler = dataset_handlers[args.dataset]

        # Instantiate the correct model builder based on the command line
        # argument
        model_builders = {
            "mnist": MNISTModelBuilder(stochastic_mode),
            "cifar10": Cifar10ModelBuilder(stochastic_mode),
            "fashion_mnist": FashionMnistModelBuilder(stochastic_mode),
        }

        model_builder = model_builders[args.dataset]

        (x_test, y_test) = dataset_handler.load_and_preprocess()

        logging.debug("Starting evaluation...")
        try:
            evaluator = Evaluator(model_builder, (x_test, y_test), args)
            evaluator.evaluate()

            send_data = SendReportData(self._report_location)
            send_data.adversarial_evaluated = args.adv_eval
            send_data.filenames = evaluator.plot_file_names

            logging.debug(
                "Evaluation complete, filenames: %s", evaluator.plot_file_names
            )

            if args.report:
                self.report()

            if args.interactive:
                send_interactive_data = SendReportData(
                    self._interactive_report_location
                )
                send_interactive_data.adversarial_evaluated = args.adv_eval
                send_interactive_data.filenames = evaluator.interactive_plot_file_names

                logging.debug(
                    "Evaluation complete, filenames: %s",
                    evaluator.interactive_plot_file_names,
                )

        except Exception as e:
            logging.error("An error occurred during evaluation: %s", e)

    def analyze(self, args):
        model_path = (
            args.model_path
            if (hasattr(args, "model_path") and args.model_path is not None)
            else input(
                "Enter the model path for analysis or press Enter to use the default path: "
            ).strip()
        )

        if not model_path:
            model_path = Analyzer.default_path

        batch = getattr(args, "batch", 64)

        # Instantiate the correct model builder based on the command line
        # argument
        dataset_handlers = {
            "mnist": MnistDatasetHandler(),
            "cifar10": Cifar10DatasetHandler(),
            "fashion_mnist": FashionMnistDatasetHandler(),
        }

        # Instantiate the correct model builder based on the command line
        # argument
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

            SendReportData(self._report_location).filenames = analyzer.plot_file_names
            if args.report:
                self.report()

            if args.interactive:
                SendReportData(self._interactive_report_location).filenames = (
                    analyzer.interactive_plot_file_names
                )
                # self.reportInteractive()

        except Exception as e:
            logging.error("An error occurred during uncertainty analysis: %s", e)

    def report(self, args=""):
        """Run report generation"""
        if args.interactive:
            send_data = SendReportData(self._interactive_report_location)
        else:
            send_data = SendReportData(self._report_location)

        if args and hasattr(args, "adv_eval"):
            send_data.adversarial_evaluated = args.adv_eval
        else:
            send_data.adversarial_evaluated = False

        try:
            if args.interactive:
                send_data.send("./src/report/reports/", "index_interactive")
            else:
                send_data.send()
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
