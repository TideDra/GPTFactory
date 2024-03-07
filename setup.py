try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

VERSION = '0.2.0'

setup(
    name='GPTFactory',
    version=VERSION,
    author='TideDra',
    author_email='gearyzhang@outlook.com',
    url='https://github.com/TideDra/GPTFactory',
    description='An all-in-one pipeline that collects data from ChatGPT models.',
    keywords=['ChatGPT','OpenAI','data collection','pipeline'],
    packages=['GPTFactory'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Utilities'
        'License :: OSI Approved :: MIT License'
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    install_requires=['requests','rich','pillow'],
    python_requires='>=3.5',
    license='MIT license',
)
