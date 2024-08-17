from setuptools import setup, find_packages

setup(
    name='SqueakyCleanText',
    version='0.2.2',
    author='Rehan Fazal',
    description='A comprehensive text cleaning and preprocessing pipeline.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rhnfzl/SqueakyCleanText',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'lingua-language-detector>=2.0.2',
        'nltk>=3.8',
        'emoji>=2.8',
        'ftfy>=6.1',
        'Unidecode>=1.3',
        'beautifulsoup4>=4.12',
        'transformers>=4.30',
        'torch>=2.0.0',
        'presidio_anonymizer>=2.2.355',
    ],
    extras_require={
        'dev': [
            'hypothesis==6.82.7',
            'faker==20.1.0',
            'flake8==6.1.0',
            'pytest==7.5.0',
        ],
        'test': [
            'coverage==7.3.1',
            'pytest-cov==4.1.0',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries',
        'Topic :: Text Processing',
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'nltk_downloader=sct.scripts.download_nltk_stopwords:main'
        ],
    },
    test_suite='tests',
    keywords='text cleaning, text preprocessing, NLP, natural language processing',
    package_data={
        '': ['*.txt', '*.rst', '*.md'],
    },
)