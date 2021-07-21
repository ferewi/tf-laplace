from setuptools import setup
from setuptools import find_packages

setup(name='tf-laplace',
      version='1.0',
      description='Scalable Laplace Approximation for estimating the posterior distribution of neural networks for '
                  'TensorFlow 2.x models.',
      author='Ferdinand Rewicki',
      author_email='ferdinand.rewicki@gmail.com',
      license='Public',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'tensorflow',
            'tensorflow==2.5',
            'tensorflow_probability'
      ]
)
