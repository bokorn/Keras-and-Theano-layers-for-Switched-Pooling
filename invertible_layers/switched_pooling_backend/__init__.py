import os
import json
import sys

_keras_base_dir = os.path.expanduser('~')
if not os.access(_keras_base_dir, os.W_OK):
    _keras_base_dir = '/tmp'

_keras_dir = os.path.join(_keras_base_dir, '.keras')
if not os.path.exists(_keras_dir):
    os.makedirs(_keras_dir)

_BACKEND = 'theano'
_config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
if os.path.exists(_config_path):
    _config = json.load(open(_config_path))
    _backend = _config.get('backend', _BACKEND)
    assert _backend in {'theano', 'tensorflow'}

    _BACKEND = _backend


if 'KERAS_BACKEND' in os.environ:
    _backend = os.environ['KERAS_BACKEND']
    assert _backend in {'theano', 'tensorflow'}
    _BACKEND = _backend

if _BACKEND == 'theano':
    sys.stderr.write('Using Theano Switched Pooling backend.\n')
    from .theano_switched_pooling_backend import *
elif _BACKEND == 'tensorflow':
    sys.stderr.write('Using TensorFlow Switched Pooling backend.\n')
    from .tensorflow_switched_pooling_backend import *
else:
    raise Exception('Unknown backend: ' + str(_BACKEND))
