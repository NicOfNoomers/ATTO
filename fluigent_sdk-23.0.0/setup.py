from setuptools import setup, find_packages

from Fluigent.SDK import __version__

setup(name="fluigent_sdk",
      version=__version__,
      description="SDK for Fluigent Instruments",
      url="http://www.fluigent.com",
      author="Fluigent",
      author_email="support@fluigent.com",
      license="Proprietary",
      packages=find_packages(exclude=("tests",)),
      namespace_packages=["Fluigent"],
      package_data={"Fluigent.SDK": ["shared/windows/*/*.dll",
                                     "shared/linux/*/*.so",
                                     "shared/mac/*/*.dylib"]},
      zip_safe=False)
