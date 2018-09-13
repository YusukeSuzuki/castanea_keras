from setuptools import setup, find_packages

setup(
    name = 'castanea_keras',
    author = 'yusuke@geekfield.jp',
    version = 0.1,
    packages = find_packages(),
    install_requires = ['tensorflow-gpu', 'numpy', 'PyYAML', 'Keras', 'keras-preprocessing', 'pillow'],
    zip_safe=False)

