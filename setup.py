from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='hier_class',
    version='0.0.1',
    description='Hierarchical Classifier',
    long_description=readme,
    packages=find_packages(exclude=(
            'logs', 'saved', 'data')),
    install_requires=reqs.strip().split('\n'),
)
