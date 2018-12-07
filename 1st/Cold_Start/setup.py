import os
from setuptools import setup, find_packages

#Version of the software
#from coldstart import __version__

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="coldstart",
    version="0.1.0",
    author="ironbar",
    author_email="guillermobarbadillo@gmail.com",
    description=("forecasting the global energy consumption of a building"),
    license='All rights reserved',
    long_description=read('README.md'),
    classifiers=[],
#    packages=find_packages("coldstart", exclude=['notebooks', 'reports', 'tests',
#                                    'logs', 'forum', 'data']),
   
    packages=find_packages(),
#    package_dir={'': "coldstart"}, 
    include_package_data=True,
    zip_safe=False,
    setup_requires=['pytest-runner',],
    tests_require=['pytest',],
    test_suite='tests',
)
