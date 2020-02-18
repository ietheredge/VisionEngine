from setuptools import setup

setup(
    name='VisionEngine',
    version='0.1',
    description='VisionEngine is a machine learning framework for analyzing \
        natural color patterns',
    url='http://github.com/ietheredge/VisionEngine',
    author='R. Ian Etherdge',
    author_email='ietheredge@ab.mpg.de',
    license='MIT',
    packages=['VisionEngine', 'VisionEngine.datasets'],
    zip_safe=False,
    package_dir={
        'VisionEngine': 'src',
        'VisionEngine.datasets': 'src/data/datasets',
        }
)
