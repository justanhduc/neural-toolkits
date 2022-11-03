import sys
from setuptools import setup, find_packages
import os
import versioneer

cwd = os.getcwd()
version_data = versioneer.get_versions()
CMD_CLASS = versioneer.get_cmdclass()

if version_data['error'] is not None:
    # Get the fallback version
    # We can't import neural_toolkits.version as it isn't yet installed, so parse it.
    fname = os.path.join(os.path.split(__file__)[0], "neural_toolkits", "version.py")
    with open(fname, "r") as f:
        lines = f.readlines()
    lines = [l for l in lines if l.startswith("FALLBACK_VERSION")]
    assert len(lines) == 1

    FALLBACK_VERSION = lines[0].split("=")[1].strip().strip('""')

    version_data['version'] = FALLBACK_VERSION


def get_extensions():
    if '--cuda-ext' in sys.argv:
        import glob
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        CMD_CLASS.update({'build_ext': BuildExtension})
        ext_root = os.path.join(cwd, 'neural_toolkits/extensions')
        ext_src = glob.glob(os.path.join(ext_root, 'csrc/*.cpp')) + glob.glob(os.path.join(ext_root, 'csrc/*.cu'))
        ext_include = os.path.join(ext_root, 'include')
        sys.argv.remove("--cuda-ext")
        return [
            CUDAExtension(
                name='neural_toolkits.ext',
                sources=ext_src,
                include_dirs=[ext_include]
            )]
    else:
        return []


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

cuda_ext = get_extensions()
setup(
    name='neural-toolkits',
    version=version_data['version'],
    # version=versioneer.get_version(),
    description='A high-level library consisting of common  on top of Pytorch.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/justanhduc/neural-toolkits',
    author='Duc Nguyen',
    author_email='adnguyen@yonsei.ac.kr',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    platforms=['Windows', 'Linux'],
    packages=find_packages(exclude=['docs', 'tests', 'examples']),
    ext_modules=cuda_ext,
    cmdclass=CMD_CLASS,
    install_requires=['matplotlib', 'scipy', 'numpy', 'imageio', 'future', 'tensorboard', 'git-python'],
    extras_require={
        'gin': ['gin-config'],
        'geom': ['pykeops', 'geomloss'],

    },
    project_urls={
        'Bug Reports': 'https://github.com/justanhduc/neural-toolkits/issues',
        'Source': 'https://github.com/justanhduc/neural-toolkits',
    },
)
