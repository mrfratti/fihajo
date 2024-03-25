from report.API.ReportGenApi import ReportGenApi


class SendReportData:
    def __init__(self):
        self._filenames = []

    @property
    def filenames(self):
        if len(self._filenames) < 1:
            print("No filenames to be sent")
        else:
            print("the following filnames is in list")
            for filename in self._filenames:
                print(filename)

    @filenames.setter
    def filenames(self, filename_list):
        if not isinstance(filename_list, list):
            raise TypeError(
                "Send report: The filnames to be sent needs to be in a list"
            )
        if len(filename_list) < 1:
            raise ValueError("Send report: No filenames for images in filename list")
        self._filenames = filename_list

    def send(self, report_location="", report_filename=""):
        images = []

        for image in self._filenames:
            if image["training"]:
                images.append(
                    {
                        "image_header": "Training",
                        "image_location": f"training/{image['training']}",
                        "about_image": "lorem ipsum",
                    }
                )
        ReportGenApi(
            report_filename=report_filename,
            report_location=report_location,
            images=images,
        )
