from distutils.core import setup

from setuptools import find_packages

setup_requires = []
install_requires = [
    'numpy==1.15',
    'pandas==0.23',
    'six==1.11',
    'scikit-learn==0.19',
    'scipy==1.1'
]

setup(name='AMED miRNA project preprocessor',
      version='0.0.1',  # NOQA
      description='',
      author='Kenta Oono',
      author_email='oono@preferred.jp',
      packages=find_packages(),
      setup_requires=setup_requires,
      install_requires=install_requires
      )
