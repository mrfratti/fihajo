import json
import os
import datetime


class Interactive_Html_Data:
    def create_div_file_html(self, data_content):

        tag_start = "<body>"
        tag_end = "</body>"
        content_start = data_content.find(tag_start) + len(tag_start)
        content_end = data_content.find(tag_end)
        
        if content_start >= len(tag_start) and content_end != -1:
            return data_content[content_start:content_end].strip()
        else:
            return data_content
