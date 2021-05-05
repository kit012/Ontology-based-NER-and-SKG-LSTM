import setuptools

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='stock_prediction',
    version='1.0.0-SNAPSHOT',
    author="Kit Luk",
    author_email="tkluk6@connect.hku.hk",
    description="Using Ontology-based NER and Knowledge Graph for Stock Price Prediction",
    long_description=long_description,
    license="",
    url="",
    install_requires=[
        "PyYAML>=5.3.1",
        "sqlalchemy>=1.3.20",
        "pymysql>=0.10.0",
        "bs4>=0.0.1",
        "selenium>=3.141.0"
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ]
)
