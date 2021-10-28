import os
import platform
import re
import subprocess
import sys

from distutils.version import LooseVersion
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):  # pylint: disable=too-few-public-methods
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError as os_error:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions)) from os_error

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        import tensorflow as tf
        tf_lib_dir = tf.sysconfig.get_lib()
        tf_file = 'libtensorflow_framework.so'
        if not os.path.exists(os.path.join(tf_lib_dir, tf_file)):
            tf_cur_file = [f for f in os.listdir(tf_lib_dir) if f.startswith(tf_file)][0]
            print(f"Creating symlink {tf_file} -> {tf_cur_file}")
            os.symlink(
                os.path.join(tf_lib_dir, tf_cur_file),
                os.path.join(tf_lib_dir, tf_file),
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='tf_radon',
      version="1.0.0",
      author="Philipp Ernst",
      author_email="phil23940@yahoo.de",
      description="Radon transform in Tensorflow",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/phernst/tf-radon",
      packages=['tf_radon'],
      package_dir={
          'tf_radon': './tf_radon',
      },
      ext_modules=[CMakeExtension('.')],
      cmdclass=dict(build_ext=CMakeBuild),
      include_package_data=True,
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: POSIX :: Linux",
          "Operating System :: Microsoft :: Windows",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
      ],
)
