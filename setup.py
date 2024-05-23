from setuptools import setup, find_packages

setup(
    name='mirrorSDF',
    version='0.1.0',
    author='Guillaume Leclerc',
    author_email='leclerc@mit.edu',
    description='3D reconstruction of materials and shape from RGB images',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://github.com/GuillaumeLeclerc/mirrorSDF',
    packages=find_packages(),
    install_requires=[
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.8',
)
