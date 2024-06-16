from setuptools import setup, find_packages

setup(
    name='SqueakyCleanText',
    version='0.1.0',
    author='Rehan Fazal',
    description='A comprehensive text cleaning and preprocessing pipeline.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/SqueakyCleanText',
    license='GPL-3.0-or-later',
    packages=find_packages(),
    install_requires=[
        'lingua-language-detector',
        'nltk',
        'emoji',
        'ftfy',
        'Unidecode',
        'beautifulsoup4',
        'transformers',
        'torch',
    ],
    extras_require={
        'dev': [
            'hypothesis',
            'faker',
            'flake8',
            'pytest',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'nltk_downloader=sct.scripts.download_nltk_stopwords:main'
        ],
    },
    test_suite='tests',
)
