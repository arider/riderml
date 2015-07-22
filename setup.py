import os
from setuptools import setup, find_packages


def read(*paths):
    """Build a file path from *paths* and return the contents."""
    with open(os.path.join(*paths), 'r') as f:
        return f.read()

setup(
    name='riderml',
    version='0.1.0',
    license='MIT licence, see LICENCE.txt',
    description='Assorted machine learning algorithms',
    long_description=(read('README.md')),
    url='',
    author='Andrew Rider',
    author_email='andrew.rider',
    include_package_data=True,
    packages=find_packages(exclude=['test*']),
)
