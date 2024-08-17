from sct.utils import constants


class ProcessDateTime:
    
    def __init__(self):
        pass
    
    def replace_years(self, text, replace_with ="<YEAR>"):
        """
        Replaces years between 1900 to 2099 in the text with a special token.
        """
        cleaned_string = constants.YEAR_REGEX.sub(replace_with, text)

        return cleaned_string