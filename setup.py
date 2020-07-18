from setuptools import setup, find_packages

setup(
    name='VisionEngine',
    version='0.1a',
    description='VisionEngine is a machine learning framework for analyzing \
        natural color patterns',
    url='http://github.com/ietheredge/VisionEngine',
    author='R. Ian Etherdge',
    author_email='ietheredge@ab.mpg.de',
    license='MIT',
    packages=find_packages(include=["VisionEngine*"])
)
