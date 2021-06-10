'''
Created on May 6, 2021

@author: graflu
'''

import os
import pytest
from distutils import dir_util


@pytest.fixture
def datadir(tmpdir, request):
    '''
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.

    Taken from stackoverflow
    https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    (May 6th 2021)
    '''
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir
