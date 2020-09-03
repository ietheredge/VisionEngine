from setuptools import setup, find_packages

if __name__ == "__main__":
    with open("README.md") as f:
        long_description = f.read()

    setup(
        name='VisionEngine',
        version='0.1a',
        description='VisionEngine is a machine learning framework for analyzing \
            natural color patterns',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='http://github.com/ietheredge/VisionEngine',
        author='R. Ian Etherdge',
        author_email='ietheredge@ab.mpg.de',
        license='MIT',
        packages=find_packages(include=["VisionEngine*"])
    )
