from setuptools import setup, find_packages

setup(
    name='SqueakyCleanText',
    version='0.1.6',
    author='Rehan Fazal',
    description='A comprehensive text cleaning and preprocessing pipeline.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rhnfzl/SqueakyCleanText',
    license='MIT',
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
        'presidio_anonymizer',
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
        'License :: OSI Approved :: MIT License',
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