import warnings
warnings.filterwarnings('ignore')

from lingua import Language, LanguageDetectorBuilder

#---- Detect the Language / Also add languages to support in Future
LANGUAGES = [Language.DUTCH, Language.ENGLISH, Language.GERMAN, Language.SPANISH]
DETECTOR = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()