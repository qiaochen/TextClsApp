#!/usr/bin/env python

from setuptools import setup, Command, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='txtclsapp',
      version='0.0.1',
      description='text classification app',
      license='None',
      author='Chen Qiao',
      author_email='cqiaohku@gmail.com',
      url='https://github.com/qiaochen/TextClsApp',
      packages=find_packages(),
      install_requires = required,
      long_description= ("This is a demo app that builds on relatively stabalized ETL, NLP and ML piplines. This would be my "
                         "skeleton of future projets")
     )