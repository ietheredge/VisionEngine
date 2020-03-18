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
    packages=[
        'VisionEngine',
        'VisionEngine.datasets',
        'VisionEngine.utils',
        'VisionEngine.data_loaders',
        'VisionEngine.base',
        'VisionEngine.models',
        'VisionEngine.trainers'
        ],
    zip_safe=False,
    package_dir={
        'VisionEngine': 'VisionEngine',
        'VisionEngine.datasets': 'VisionEngine/data_loaders/datasets',
        'VisionEngine.utils': 'VisionEngine/utils',
        'VisionEngine.data_loaders': 'VisionEngine/data_loaders',
        'VisionEngine.base': 'VisionEngine/base',
        'VisionEngine.models': 'VisionEngine/models',
        'VisionEngine.trainers': 'VisionEngine/trainers'
        }
)
