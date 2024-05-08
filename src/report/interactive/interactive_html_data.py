import json
import os
import datetime


class Interactive_Html_Data:
    def create_div_file_html(self, data_content):
        html_content = ""

        content_start = data_content.index("<body>\n") + 1
        content_end = data_content.index("</body>\n")
        content_content = data_content[content_start:content_end]
        content_string = "".join(content_content)
        html_content += content_string
        
        return html_content