import unittest
from unittest.mock import patch, MagicMock
from src.cli.main import CLIApp


class TestCLIApp(unittest.TestCase):
    @patch(
        "src.cli.main.argparse.ArgumentParser.parse_args",
        return_value=MagicMock(
            config="train.json", verbose=False, quiet=False, command="train"
        ),
    )
    @patch(
        "src.cli.main.CLIApp.load_config",
        return_value={
            "command": "train",
            "dataset": "mnist",
            "epochs": 1,
            "batch": 1000,
            "adv": False,
            "optimizer": "adam",
            "learning_rate": None,
        },
    )
    @patch("src.cli.main.CLIApp.train")
    def test_run_train_command(self, mock_train):
        app = CLIApp()
        status_code = app.run()  # pylint: disable=E1111
        mock_train.assert_called_once_with(app.args)
        self.assertEqual(status_code, None)


if __name__ == "__main__":
    unittest.main()
