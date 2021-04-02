#!/usr/bin/env python
#
# Copyright (c) 2018, Oracle and/or its affiliates. All rights reserved.
#
# Licensed under the Universal Permissive License v 1.0 as shown at
# http://oss.oracle.com/licenses/upl.


import collections
import os.path
import os
import stat

import tensorflow as tf

from tensorflow.keras import models
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import convert_to_constants as convert
from tensorflow.python.keras.saving import saving_utils


def write_graph(graph_def, fname):
    d, f = os.path.split(os.path.abspath(fname))
    tf.io.write_graph(graph_or_graph_def=graph_def,
                  logdir=d,
                  name=f"{fname}",
                  as_text=False)


def constantize(fname):
    model = models.load_model(fname)

    input_signature = None
    # If the model is not a function then the model may include
    # a specific batch size, so we include it as well.
    if not isinstance(model.call, def_function.Function):
        input_signature = saving_utils.model_input_signature(
            model, keep_original_batch_size=True)

    func = saving_utils.trace_model_call(model, input_signature)
    concrete_func = func.get_concrete_function()
    _, graph_def = convert.convert_variables_to_constants_v2_as_graph(
        concrete_func, lower_control_flow=False)


    return graph_def


def h5_to_pb(h5, pb):
    write_graph(constantize(h5), pb)


def copy_perms(source, target):
    st = os.stat(source)
    os.chown(target, st[stat.ST_UID], st[stat.ST_GID])


if __name__ == "__main__":
    # disable gpu for conversion
    tf.config.set_visible_devices([], 'GPU')

    import sys
    if len(sys.argv) < 3:
        print('usage: {} <src_fname> <dst_fname>'.format(sys.argv[0]))
        sys.exit(1)
    h5_to_pb(sys.argv[1], sys.argv[2])
    copy_perms(sys.argv[1], sys.argv[2])
    print('saved the constant graph (ready for inference) at: ', sys.argv[2])
