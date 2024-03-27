class StringStyling:
    """Different ways of styling strings"""

    @staticmethod
    def box_style(message, character="#") -> str:
        """Makes an box around an message"""
        message = f"{character} {message} {character}"
        grid = "\n"
        box_length = len(message) if len(message) < 80 else 80
        for i in range(0, box_length):
            grid += character
        grid += "\n"
        message = grid + message + grid
        return message
