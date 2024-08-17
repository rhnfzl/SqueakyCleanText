import re
from nltk.corpus import stopwords as sw

class ProcessStopwords:
    
    def __init__(self):
        ### stop words
        self.STOP_WORDS_EN = sw.words('english')
        self.STOP_WORDS_NL = sw.words('dutch')
        self.STOP_WORDS_DE = sw.words('german')
        self.STOP_WORDS_ES = sw.words('spanish')

    def remove_stopwords(self, text: str, lan: str) -> str:
        """
        Removes stopwords based on the language.

        Parameters:
            text (str): The input text to have stopwords removed from.
            lan (str): The language of the text.

        Returns:
            str: The text with all stopwords removed.
        """

        # Determine the stopwords based on the language
        stop_words = {
            'DUTCH': self.STOP_WORDS_NL,
            'ENGLISH': self.STOP_WORDS_EN,
            'GERMAN': self.STOP_WORDS_DE,
            'SPANISH': self.STOP_WORDS_ES
        }.get(lan, [])

        # Split the text into words and remove stopwords
        text = text.split()
        return " ".join([word for word in text if word not in stop_words])

    def remove_words_from_string(self, text: str, words_to_remove: list[str]) -> str:
        """
        Removes a list of words from a given text.

        This function takes a string of text and a list of words to remove,
        and returns a new string with all occurrences of the words removed.

        Parameters
        ----------
        text : str
            The input text to have words removed from.
        words_to_remove : list[str]
            A list of words to remove from the text.

        Returns
        -------
        str
            The text with all occurrences of the words removed.
        """

        # Join the words with the "|" (OR) operator in regex to create a pattern
        pattern = r'\b(?:' + '|'.join(re.escape(word) for word in words_to_remove) + r')\b'
        
        # Use re.sub() to replace all matches with an empty string
        result_text = re.sub(pattern, '', text)
        return result_text
