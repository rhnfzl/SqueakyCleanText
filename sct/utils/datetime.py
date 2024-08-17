from sct.utils import constants

class ProcessDateTime:
    
    def __init__(self):
        pass
    
    def replace_years(self, text: str, replacement: str = "<YEAR>") -> str:
        """
        Replaces years between 1900 to 2099 in the text with a special token.

        Args:
            text (str): The input text.
            replacement (str, optional): The string to replace the years with.
                Defaults to "<YEAR>".

        Returns:
            str: The modified text with years replaced.
        """
        # Use the pre-compiled regular expression to find all year patterns
        # in the text and replace them with the specified replacement string

        # Substitute all year patterns with special token
        cleaned_string = constants.YEAR_REGEX.sub(replacement, text)

        return cleaned_string
