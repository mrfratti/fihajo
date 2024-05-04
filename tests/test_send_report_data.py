import unittest
from unittest.mock import patch, MagicMock, mock_open
from src.cli.send.send_report_data import SendReportData


class TestSendReportData(unittest.TestCase):
    def setUp(self):
        self.send_report_data = SendReportData()

    @patch('os.path.exists')
    @patch('os.remove')
    def test_delete_json(self, mock_remove, mock_exists):
        # Case: JSON file exists
        mock_exists.return_value = True
        self.send_report_data.delete_json()
        mock_remove.assert_called_once_with(self.send_report_data._path_json)

        # Case: JSON file does not exist
        mock_remove.reset_mock()
        mock_exists.return_value = False
        self.send_report_data.delete_json()
        mock_remove.assert_not_called()

    @patch('builtins.open', new_callable=mock_open, read_data='{"existing": "data"}')
    @patch('os.path.isfile')
    @patch('os.stat')
    def test_load_json(self, mock_stat, mock_isfile, mock_file):
        mock_isfile.return_value = True
        mock_stat.return_value.st_size = 1  # Non-zero size
        result = self.send_report_data._load_json()
        self.assertEqual(result, {"existing": "data"})
        mock_file.assert_called_once_with(self.send_report_data._path_json, "r", encoding="UTF-8")

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_filenames_setter(self, mock_json_dump, mock_file):
        self.send_report_data._filenames = {}
        self.send_report_data.filenames = {'new': 'data'}
        mock_file.assert_called_once_with(self.send_report_data._path_json, "w", encoding="UTF-8")
        mock_json_dump.assert_called_once()

    @patch('builtins.print')
    def test_filenames_getter(self, mock_print):
        # Case: No filenames
        self.send_report_data._filenames = {}
        self.send_report_data.filenames
        mock_print.assert_any_call("No filenames to be sent")

        # Case: Some filenames
        mock_print.reset_mock()
        self.send_report_data._filenames = {'file1': 'path1', 'file2': 'path2'}
        self.send_report_data.filenames
        mock_print.assert_any_call("the following filenames is in list")

    def test_adversarial_evaluated_setter_getter(self):
        self.assertFalse(self.send_report_data.adversarial_evaluated)
        self.send_report_data.adversarial_evaluated = True
        self.assertTrue(self.send_report_data.adversarial_evaluated)

    @patch('src.cli.send.send_report_data.HtmlGeneratorApi')
    def test_send(self, mock_html_generator_api):
        self.send_report_data._filenames = {'key1': 'file1'}
        with patch.object(self.send_report_data, '_load_json', return_value=self.send_report_data._filenames):
            self.send_report_data.send(report_location='loc', report_filename='file')
            mock_html_generator_api.assert_called_once()

    @patch('src.cli.send.send_report_data.HtmlGeneratorApi')
    def test_send_no_images(self, mock_html_generator_api):
        with patch.object(self.send_report_data, '_load_json', return_value={}):
            with self.assertRaises(ValueError):
                self.send_report_data.send()


if __name__ == '__main__':
    unittest.main()
