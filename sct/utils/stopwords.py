import re
#---
from sct.utils import resources
from nltk.corpus import stopwords as sw

class ProcessStopwords:
    
    def __init__(self):
        ### stop words
        self.STOP_WORDS_EN = sw.words('english')
        self.STOP_WORDS_NL = sw.words('dutch')
        self.STOP_WORDS_DE = sw.words('german')

    def remove_stopwords(self, text, lan):
        """
        Removes stopwords based on the language.
        """
        if lan == 'DUTCH':
            stop_words = self.STOP_WORDS_NL
        elif lan == 'ENGLISH':
            stop_words = self.STOP_WORDS_EN
        elif lan == 'GERMAN':
            stop_words = self.STOP_WORDS_DE

        text = text.split()
        return " ".join([word for word in text if word not in stop_words])

    def remove_words_from_string(self, text, words_to_remove):
        # Join the words with the "|" (OR) operator in regex to create a pattern
        pattern = r'\b(?:' + '|'.join(re.escape(word) for word in words_to_remove) + r')\b'
        
        # Use re.sub() to replace all matches with an empty string
        result_text = re.sub(pattern, '', text)
        return result_text