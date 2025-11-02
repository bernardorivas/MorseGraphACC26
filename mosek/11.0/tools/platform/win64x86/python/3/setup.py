from setuptools import Extension
from setuptools import setup
import logging
import pathlib
import platform
import setuptools
import setuptools.command.build_ext
import setuptools.command.install
import shutil
import subprocess
import sys,os,re

class InstallationError(Exception): pass

major,minor,_,_,_ = sys.version_info
setupdir = pathlib.Path(__file__).resolve().parent

python_versions = [(3, 9), (3, 10), (3, 11), (3, 12), (3, 13)]
if (major,minor) not in python_versions: raise InstallationError("Unsupported python version")


class build_ext(setuptools.command.build_ext.build_ext):
    """
    Extend the default `build_ext` command replacing the extension
    building functionality with one that simply copies a pre-built
    extension module.
    """
    def build_extension(self,ext):
        tgtdir = pathlib.Path(self.build_lib).joinpath(*ext.name.split('.')[:-1])

        try: os.makedirs(tgtdir)
        except OSError: pass
        for s in ext.sources:
            logging.info("copying %s -> %s" % (s,tgtdir))
            shutil.copy(s,tgtdir)

class install(setuptools.command.install.install):
    """
    Extend the default install command, adding an additional operation
    that installs the dynamic MOSEK libraries.
    """
    libdir   = ['..\\..\\bin']
    instlibs = [('tbb12.dll', 'tbb12.dll'), ('mosek64_11_0.dll', 'mosek64_11_0.dll')]
    
    def findlib(self,lib):
        for p in self.libdir:
            f = pathlib.Path(p).joinpath(lib)
            if f.exists():
                return f
        raise InstallationError(f"Library not found: {lib}")
    
    def install_libs(self):
        mskdir = pathlib.Path(self.install_lib).joinpath('mosek')
        for lib,tgtname in [ (self.findlib(lib),t) for (lib,t) in self.instlibs ]:
            logging.info(f"copying {lib} -> {mskdir}")
            shutil.copy(lib,mskdir)
    def run(self):
        super().run()
        self.execute(self.install_libs, (), msg="Installing native libraries")

os.chdir(setupdir)
setup(name =        'Mosek',
      version =     '11.0.27',
      packages =    ['mosek', 'mosek.fusion', 'mosek.fusion.impl'],
      ext_modules = [ Extension("mosek._msk",sources=[r"mosek\_msk.cp39-win_amd64.pyd",r"mosek\_msk.cp310-win_amd64.pyd",r"mosek\_msk.cp311-win_amd64.pyd",r"mosek\_msk.cp312-win_amd64.pyd",r"mosek\_msk.cp313-win_amd64.pyd"]),Extension("mosek.fusion.impl.fragments",sources=[r"mosek\fusion\impl\fragments.cp39-win_amd64.pyd",r"mosek\fusion\impl\fragments.cp310-win_amd64.pyd",r"mosek\fusion\impl\fragments.cp311-win_amd64.pyd",r"mosek\fusion\impl\fragments.cp312-win_amd64.pyd",r"mosek\fusion\impl\fragments.cp313-win_amd64.pyd"]) ],
      cmdclass =    { "build_ext" : build_ext, "install" : install })
