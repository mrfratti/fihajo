class String_Format:
    def message(message):
        message = f"# {message} #"
        grid = "\n"
        box_length = len(message) if len(message) < 80 else 80
        for i in range(0, box_length):
            grid += "#"
        grid += "\n"
        message = grid + message + grid
        return message
