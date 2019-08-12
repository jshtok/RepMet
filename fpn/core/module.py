# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Zheng Zhang
# --------------------------------------------------------

"""A `MutableModule` implement the `BaseModule` API, and allows input shape
varying with training iterations. If shapes vary, executors will rebind,
using shared arrays from the initial module binded with maximum shape.
"""

import time
import logging
import warnings
import random
import cPickle
import copy
from mxnet import context as ctx
from mxnet.initializer import Uniform, InitDesc
from mxnet.module.base_module import BaseModule, _check_input_names, _parse_data_desc, _as_list
from mxnet.model import _create_kvstore, _initialize_kvstore, _update_params, _update_params_on_kvstore, load_checkpoint, BatchEndParam
from mxnet import metric
from easydict import EasyDict as edict
from utils.tfs_vis import vis_reps_TSNE, normalize_reps
from utils.miscellaneous import assert_folder
from core import callback
# from mxnet.module.executor_group import DataParallelExecutorGroup
import numpy as np
import os
from .DataParallelExecutorGroup import DataParallelExecutorGroup
from mxnet import ndarray as nd
from mxnet import optimizer as opt
from bbox.bbox_utils import bb_overlap
from sklearn.cluster import KMeans
#from mxboard import SummaryWriter ################### mxboard

LEONID_PROFILING_ENABLED = False
JS_DEBUG = False

class Module(BaseModule):
    """Module is a basic module that wrap a `Symbol`. It is functionally the same
    as the `FeedForward` model, except under the module API.

    Parameters
    ----------
    symbol : Symbol
    data_names : list of str
        Default is `('data')` for a typical model used in image classification.
    label_names : list of str
        Default is `('softmax_label')` for a typical model used in image
        classification.
    logger : Logger
        Default is `logging`.
    context : Context or list of Context
        Default is `cpu()`.
    work_load_list : list of number
        Default `None`, indicating uniform workload.
    fixed_param_names: list of str
        Default `None`, indicating no network parameters are fixed.
    state_names : list of str
        states are similar to data and label, but not provided by data iterator.
        Instead they are initialized to 0 and can be set by set_states()
    """
    def __init__(self, symbol, data_names=('data',), label_names=('softmax_label',),
                 logger=logging, context=ctx.cpu(), work_load_list=None,
                 fixed_param_names=None, alt_fixed_param_names=None, state_names=None):
        super(Module, self).__init__(logger=logger)

        if isinstance(context, ctx.Context):
            context = [context]
        self._context = context
        if work_load_list is None:
            work_load_list = [1] * len(self._context)
        assert len(work_load_list) == len(self._context)
        self._work_load_list = work_load_list

        self._symbol = symbol

        data_names = list(data_names) if data_names is not None else []
        label_names = list(label_names) if label_names is not None else []
        state_names = list(state_names) if state_names is not None else []
        fixed_param_names = list(fixed_param_names) if fixed_param_names is not None else []
        alt_fixed_param_names = list(alt_fixed_param_names) if alt_fixed_param_names is not None else []

        _check_input_names(symbol, data_names, "data", True)
        _check_input_names(symbol, label_names, "label", False)
        _check_input_names(symbol, state_names, "state", True)
        _check_input_names(symbol, fixed_param_names, "fixed_param", True)
        _check_input_names(symbol, alt_fixed_param_names, "alt_fixed_param", True)

        arg_names = symbol.list_arguments()
        input_names = data_names + label_names + state_names
        self._param_names = [x for x in arg_names if x not in input_names]
        self._fixed_param_names = fixed_param_names
        self._alt_fixed_param_names = alt_fixed_param_names
        self._aux_names = symbol.list_auxiliary_states()
        self._data_names = data_names
        self._label_names = label_names
        self._state_names = state_names
        self._output_names = symbol.list_outputs()

        self._arg_params = None
        self._aux_params = None
        self._params_dirty = False

        self._optimizer = None
        self._kvstore = None
        self._update_on_kvstore = None
        self._updater = None
        self._preload_opt_states = None
        self._grad_req = None

        self._exec_group = None
        self._data_shapes = None
        self._label_shapes = None

    @staticmethod
    def load(prefix, epoch, load_optimizer_states=False, **kwargs):
        """Create a model from previously saved checkpoint.

        Parameters
        ----------
        prefix : str
            path prefix of saved model files. You should have
            "prefix-symbol.json", "prefix-xxxx.params", and
            optionally "prefix-xxxx.states", where xxxx is the
            epoch number.
        epoch : int
            epoch to load.
        load_optimizer_states : bool
            whether to load optimizer states. Checkpoint needs
            to have been made with save_optimizer_states=True.
        data_names : list of str
            Default is `('data')` for a typical model used in image classification.
        label_names : list of str
            Default is `('softmax_label')` for a typical model used in image
            classification.
        logger : Logger
            Default is `logging`.
        context : Context or list of Context
            Default is `cpu()`.
        work_load_list : list of number
            Default `None`, indicating uniform workload.
        fixed_param_names: list of str
            Default `None`, indicating no network parameters are fixed.
        """
        sym, args, auxs = load_checkpoint(prefix, epoch)
        mod = Module(symbol=sym, **kwargs)
        mod._arg_params = args
        mod._aux_params = auxs
        mod.params_initialized = True
        if load_optimizer_states:
            mod._preload_opt_states = '%s-%04d.states'%(prefix, epoch)
        return mod

    def save_checkpoint(self, prefix, epoch, save_optimizer_states=False):
        """Save current progress to checkpoint.
        Use mx.callback.module_checkpoint as epoch_end_callback to save during training.

        Parameters
        ----------
        prefix : str
            The file prefix to checkpoint to
        epoch : int
            The current epoch number
        save_optimizer_states : bool
            Whether to save optimizer states for continue training
        """
        self._symbol.save('%s-symbol.json'%prefix)
        param_name = '%s-%04d.params' % (prefix, epoch)
        self.save_params(param_name)
        logging.info('Saved checkpoint to \"%s\"', param_name)
        if save_optimizer_states:
            state_name = '%s-%04d.states' % (prefix, epoch)
            self.save_optimizer_states(state_name)
            logging.info('Saved optimizer state to \"%s\"', state_name)

    def _reset_bind(self):
        """Internal function to reset binded state."""
        self.binded = False
        self._exec_group = None
        self._data_shapes = None
        self._label_shapes = None

    @property
    def data_names(self):
        """A list of names for data required by this module."""
        return self._data_names

    @property
    def label_names(self):
        """A list of names for labels required by this module."""
        return self._label_names

    @property
    def output_names(self):
        """A list of names for the outputs of this module."""
        return self._output_names

    @property
    def data_shapes(self):
        """Get data shapes.
        Returns
        -------
        A list of `(name, shape)` pairs.
        """
        assert self.binded
        return self._data_shapes

    @property
    def label_shapes(self):
        """Get label shapes.
        Returns
        -------
        A list of `(name, shape)` pairs. The return value could be `None` if
        the module does not need labels, or if the module is not binded for
        training (in this case, label information is not available).
        """
        assert self.binded
        return self._label_shapes

    @property
    def output_shapes(self):
        """Get output shapes.
        Returns
        -------
        A list of `(name, shape)` pairs.
        """
        assert self.binded
        return self._exec_group.get_output_shapes()

    def get_params(self):
        """Get current parameters.
        Returns
        -------
        `(arg_params, aux_params)`, each a dictionary of name to parameters (in
        `NDArray`) mapping.
        """
        assert self.binded and self.params_initialized

        if self._params_dirty:
            self._sync_params_from_devices()
        return (self._arg_params, self._aux_params)

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False, allow_extra=False):
        """Initialize the parameters and auxiliary states.

        Parameters
        ----------
        initializer : Initializer
            Called to initialize parameters if needed.
        arg_params : dict
            If not None, should be a dictionary of existing arg_params. Initialization
            will be copied from that.
        aux_params : dict
            If not None, should be a dictionary of existing aux_params. Initialization
            will be copied from that.
        allow_missing : bool
            If true, params could contain missing values, and the initializer will be
            called to fill those missing params.
        force_init : bool
            If true, will force re-initialize even if already initialized.
        """
        if self.params_initialized and not force_init:
            warnings.warn("Parameters already initialized and force_init=False. "
                          "init_params call ignored.", stacklevel=2)
            return
        assert self.binded, 'call bind before initializing the parameters'

        def _impl(name, arr, cache):
            """Internal helper for parameter initialization"""
            if cache is not None:
                if name in cache:
                    cache_arr = cache[name]

                    # just in case the cached array is just the target itself
                    if cache_arr is not arr:
                        cache_arr.copyto(arr)
                else:
                    if (not allow_missing) and ('const_eq_' not in name):
                        raise RuntimeError("%s is not presented" % name)
                    if initializer != None:
                        initializer(name, arr)
            else:
                initializer(name, arr)

        attrs = self._symbol.attr_dict()
        for name, arr in self._arg_params.items():
            desc = InitDesc(name, attrs.get(name, None))
            _impl(desc, arr, arg_params)

        for name, arr in self._aux_params.items():
            desc = InitDesc(name, attrs.get(name, None))
            _impl(desc, arr, aux_params)

        self.params_initialized = True
        self._params_dirty = False

        # copy the initialized parameters to devices
        self._exec_group.set_params(self._arg_params, self._aux_params)

    def set_params(self, arg_params, aux_params, allow_missing=False, force_init=True):
        """Assign parameter and aux state values.

        Parameters
        ----------
        arg_params : dict
            Dictionary of name to value (`NDArray`) mapping.
        aux_params : dict
            Dictionary of name to value (`NDArray`) mapping.
        allow_missing : bool
            If true, params could contain missing values, and the initializer will be
            called to fill those missing params.
        force_init : bool
            If true, will force re-initialize even if already initialized.

        Examples
        --------
        An example of setting module parameters::
            >>> sym, arg_params, aux_params = \
            >>>     mx.model.load_checkpoint(model_prefix, n_epoch_load)
            >>> mod.set_params(arg_params=arg_params, aux_params=aux_params)
        """
        if not allow_missing:
            self.init_params(initializer=None, arg_params=arg_params, aux_params=aux_params,
                             allow_missing=allow_missing, force_init=force_init)
            return

        if self.params_initialized and not force_init:
            warnings.warn("Parameters already initialized and force_init=False. "
                          "set_params call ignored.", stacklevel=2)
            return

        self._exec_group.set_params(arg_params, aux_params)

        # because we didn't update self._arg_params, they are dirty now.
        self._params_dirty = True
        self.params_initialized = True

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None,
             grad_req='write'):
        """Bind the symbols to construct executors. This is necessary before one
        can perform computation with the module.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is `data_iter.provide_data`.
        label_shapes : list of (str, tuple)
            Typically is `data_iter.provide_label`.
        for_training : bool
            Default is `True`. Whether the executors should be bind for training.
        inputs_need_grad : bool
            Default is `False`. Whether the gradients to the input data need to be computed.
            Typically this is not needed. But this might be needed when implementing composition
            of modules.
        force_rebind : bool
            Default is `False`. This function does nothing if the executors are already
            binded. But with this `True`, the executors will be forced to rebind.
        shared_module : Module
            Default is `None`. This is used in bucketing. When not `None`, the shared module
            essentially corresponds to a different bucket -- a module with different symbol
            but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
        """
        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True
        self._grad_req = grad_req

        if not for_training:
            assert not inputs_need_grad
        else:
            pass
            # this is not True, as some module might not contains a loss function
            # that consumes the labels
            # assert label_shapes is not None

        # self._data_shapes, self._label_shapes = _parse_data_desc(
        #     self.data_names, self.label_names, data_shapes, label_shapes)
        self._data_shapes, self._label_shapes = zip(*[_parse_data_desc(self.data_names, self.label_names, data_shape, label_shape)
                                                      for data_shape, label_shape in zip(data_shapes, label_shapes)])
        if self._label_shapes.count(None) == len(self._label_shapes):
            self._label_shapes = None

        if shared_module is not None:
            assert isinstance(shared_module, Module) and \
                    shared_module.binded and shared_module.params_initialized
            shared_group = shared_module._exec_group
        else:
            shared_group = None
        self._exec_group = DataParallelExecutorGroup(self._symbol, self._context,
                                                     self._work_load_list, self._data_shapes,
                                                     self._label_shapes, self._param_names,
                                                     for_training, inputs_need_grad,
                                                     shared_group, logger=self.logger,
                                                     fixed_param_names=self._fixed_param_names,
                                                     grad_req=grad_req,
                                                     state_names=self._state_names)
        # self._total_exec_bytes = self._exec_group._total_exec_bytes
        if shared_module is not None:
            self.params_initialized = True
            self._arg_params = shared_module._arg_params
            self._aux_params = shared_module._aux_params
        elif self.params_initialized:
            # if the parameters are already initialized, we are re-binding
            # so automatically copy the already initialized params
            self._exec_group.set_params(self._arg_params, self._aux_params)
        else:
            assert self._arg_params is None and self._aux_params is None
            param_arrays = [
                nd.zeros(x[0].shape, dtype=x[0].dtype)
                for x in self._exec_group.param_arrays
            ]
            self._arg_params = {name:arr for name, arr in zip(self._param_names, param_arrays)}

            aux_arrays = [
                nd.zeros(x[0].shape, dtype=x[0].dtype)
                for x in self._exec_group.aux_arrays
            ]
            self._aux_params = {name:arr for name, arr in zip(self._aux_names, aux_arrays)}

        if shared_module is not None and shared_module.optimizer_initialized:
            self.borrow_optimizer(shared_module)


    def reshape(self, data_shapes, label_shapes=None):
        """Reshape the module for new input shapes.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is `data_iter.provide_data`.
        label_shapes : list of (str, tuple)
            Typically is `data_iter.provide_label`.
        """
        assert self.binded
        # self._data_shapes, self._label_shapes = _parse_data_desc(
        #     self.data_names, self.label_names, data_shapes, label_shapes)
        self._data_shapes, self._label_shapes = zip(*[_parse_data_desc(self.data_names, self.label_names, data_shape, label_shape)
                                                      for data_shape, label_shape in zip(data_shapes, label_shapes)])

        self._exec_group.reshape(self._data_shapes, self._label_shapes)


    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        """Install and initialize optimizers.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default `False`, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        assert self.binded and self.params_initialized

        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring...')
            return

        (kvstore, update_on_kvstore) = \
                _create_kvstore(kvstore, len(self._context), self._arg_params)

        batch_size = self._exec_group.batch_size
        if kvstore and 'dist' in kvstore.type and '_sync' in kvstore.type:
            batch_size *= kvstore.num_workers
        rescale_grad = 1.0/batch_size

        if isinstance(optimizer, str):
            idx2name = {}
            if update_on_kvstore:
                idx2name.update(enumerate(self._exec_group.param_names))
            else:
                for k in range(len(self._context)):
                    idx2name.update({i*len(self._context)+k: n
                                     for i, n in enumerate(self._exec_group.param_names)})
            optimizer_params = dict(optimizer_params)
            if 'rescale_grad' not in optimizer_params:
                optimizer_params['rescale_grad'] = rescale_grad
            optimizer = opt.create(optimizer,
                                   sym=self.symbol, param_idx2name=idx2name,
                                   **optimizer_params)
        else:
            assert isinstance(optimizer, opt.Optimizer)
            if optimizer.rescale_grad != rescale_grad:
                #pylint: disable=no-member
                warnings.warn(
                    "Optimizer created manually outside Module but rescale_grad " +
                    "is not normalized to 1.0/batch_size/num_workers (%s vs. %s). "%(
                        optimizer.rescale_grad, rescale_grad) +
                    "Is this intended?", stacklevel=2)

        self._optimizer = optimizer
        self._kvstore = kvstore
        self._update_on_kvstore = update_on_kvstore
        self._updater = None

        if kvstore:
            # copy initialized local parameters to kvstore
            _initialize_kvstore(kvstore=kvstore,
                                param_arrays=self._exec_group.param_arrays,
                                arg_params=self._arg_params,
                                param_names=self._param_names,
                                update_on_kvstore=update_on_kvstore)
        if update_on_kvstore:
            kvstore.set_optimizer(self._optimizer)
        else:
            self._updater = opt.get_updater(optimizer)

        self.optimizer_initialized = True

        if self._preload_opt_states is not None:
            self.load_optimizer_states(self._preload_opt_states)
            self._preload_opt_states = None

    def borrow_optimizer(self, shared_module):
        """Borrow optimizer from a shared module. Used in bucketing, where exactly the same
        optimizer (esp. kvstore) is used.

        Parameters
        ----------
        shared_module : Module
        """
        assert shared_module.optimizer_initialized
        self._optimizer = shared_module._optimizer
        self._kvstore = shared_module._kvstore
        self._update_on_kvstore = shared_module._update_on_kvstore
        self._updater = shared_module._updater
        self.optimizer_initialized = True

    def forward(self, data_batch, is_train=None):
        """Forward computation.

        Parameters
        ----------
        data_batch : DataBatch
            Could be anything with similar API implemented.
        is_train : bool
            Default is `None`, which means `is_train` takes the value of `self.for_training`.
        """
        assert self.binded and self.params_initialized
        self._exec_group.forward(data_batch, is_train)

    def backward(self, out_grads=None):
        """Backward computation.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.
        """
        assert self.binded and self.params_initialized
        self._exec_group.backward(out_grads=out_grads)

    def update(self):
        """Update parameters according to the installed optimizer and the gradients computed
        in the previous forward-backward batch.
        """
        assert self.binded and self.params_initialized and self.optimizer_initialized

        self._params_dirty = True
        if self._update_on_kvstore:
            try:
                _update_params_on_kvstore(self._exec_group.param_arrays,
                                          self._exec_group.grad_arrays,
                                          self._kvstore)
            except:
                _update_params_on_kvstore(self._exec_group.param_arrays,
                                          self._exec_group.grad_arrays,
                                          self._kvstore, param_names=self._exec_group.param_names)
        else:
            ################# Leonid layer fix alternative ###################
            ga_copy=[x for x in self._exec_group.grad_arrays]
            for iP, P in enumerate(self._exec_group.param_names):
                if P in self._alt_fixed_param_names:
                    ga_copy[iP] = [None]
            ################# Leonid layer fix alternative ###################
            _update_params(self._exec_group.param_arrays,
                           ga_copy, #self._exec_group.grad_arrays, #changed by Leonid
                           updater=self._updater,
                           num_device=len(self._context),
                           kvstore=self._kvstore)

    def get_outputs(self, merge_multi_context=True):
        """Get outputs of the previous forward computation.

        Parameters
        ----------
        merge_multi_context : bool
            Default is `True`. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is `True`, it is like `[out1, out2]`. Otherwise, it
        is like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`. All the output
        elements are `NDArray`.
        """
        assert self.binded and self.params_initialized
        return self._exec_group.get_outputs(merge_multi_context=merge_multi_context)

    def get_input_grads(self, merge_multi_context=True):
        """Get the gradients with respect to the inputs of the module.

        Parameters
        ----------
        merge_multi_context : bool
            Default is `True`. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is `True`, it is like `[grad1, grad2]`. Otherwise, it
        is like `[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]`. All the output
        elements are `NDArray`.
        """
        assert self.binded and self.params_initialized and self.inputs_need_grad
        return self._exec_group.get_input_grads(merge_multi_context=merge_multi_context)

    def get_states(self, merge_multi_context=True):
        """Get states from all devices

        Parameters
        ----------
        merge_multi_context : bool
            Default is `True`. In the case when data-parallelism is used, the states
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is `True`, it is like `[out1, out2]`. Otherwise, it
        is like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`. All the output
        elements are `NDArray`.
        """
        assert self.binded and self.params_initialized
        return self._exec_group.get_states(merge_multi_context=merge_multi_context)

    def set_states(self, states=None, value=None):
        """Set value for states. Only one of states & value can be specified.

        Parameters
        ----------
        states : list of list of NDArrays
            source states arrays formatted like [[state1_dev1, state1_dev2],
            [state2_dev1, state2_dev2]].
        value : number
            a single scalar value for all state arrays.
        """
        assert self.binded and self.params_initialized
        self._exec_group.set_states(states, value)

    def update_metric(self, eval_metric, labels):
        """Evaluate and accumulate evaluation metric on outputs of the last forward computation.

        Parameters
        ----------
        eval_metric : EvalMetric
        labels : list of NDArray
            Typically `data_batch.label`.
        """
        self._exec_group.update_metric(eval_metric, labels)

    def _sync_params_from_devices(self):
        """Synchronize parameters from devices to CPU. This function should be called after
        calling `update` that updates the parameters on the devices, before one can read the
        latest parameters from `self._arg_params` and `self._aux_params`.
        """
        self._exec_group.get_params(self._arg_params, self._aux_params)
        self._params_dirty = False

    def save_optimizer_states(self, fname):
        """Save optimizer (updater) state to file

        Parameters
        ----------
        fname : str
            Path to output states file.
        """
        assert self.optimizer_initialized

        if self._update_on_kvstore:
            self._kvstore.save_optimizer_states(fname)
        else:
            with open(fname, 'wb') as fout:
                fout.write(self._updater.get_states())

    def load_optimizer_states(self, fname):
        """Load optimizer (updater) state from file

        Parameters
        ----------
        fname : str
            Path to input states file.
        """
        assert self.optimizer_initialized

        if self._update_on_kvstore:
            self._kvstore.load_optimizer_states(fname)
        else:
            self._updater.set_states(open(fname, 'rb').read())

    def install_monitor(self, mon):
        """ Install monitor on all executors """
        assert self.binded
        self._exec_group.install_monitor(mon)


def dictCompare(d1, d2):
    diff_items = {k: np.max(np.abs(d1[k].asnumpy() - d2[k].asnumpy())) for k in d1 if
                  (k in d2) and np.array_equal(d1[k].asnumpy().shape, d2[k].asnumpy().shape) and not np.array_equal(d1[k].asnumpy(), d2[k].asnumpy())}
    diff_items_extra1 = {k: d1[k] for k in d1 if k not in d2}
    diff_items_extra2 = {k: d2[k] for k in d2 if k not in d1}
    diff_items_extra3 = {k: (d1[k].asnumpy().shape, d2[k].asnumpy().shape) for k in d1 if (k in d2) and not np.array_equal(d1[k].asnumpy().shape, d2[k].asnumpy().shape)}
    print(diff_items)
    print(diff_items_extra1.keys())
    print(diff_items_extra2.keys())
    print(diff_items_extra3)
    return diff_items, diff_items_extra1, diff_items_extra2, diff_items_extra3


def get_reps_stats(cls_reps,cls_feats,config,logger):
    Nreps = cls_reps.shape[1]
    from utils.miscellaneous import cos_sim_2_dist_generic
    cos_sim = np.dot(cls_reps.transpose(), cls_feats)# dims=[#reps, #samples]
    # assume all is normalized.
    nearest_smpl_sim = np.max(cos_sim, axis=1)
    coverage = np.max(cos_sim, axis=0)
    coverage_mean = np.mean(coverage)
    coverage_std = np.std(coverage)
    coverage_worst = np.min(coverage)
    #all_cls_rep_dist = cos_sim_2_dist_generic(cos_sim,config,embed=cls_feats,reps=cls_reps)
    coverage_which = np.argmax(cos_sim, axis=0)
    cover_size = np.zeros((Nreps))
    for i in range(Nreps):
        cover_size[i] = int(np.where(coverage_which==i)[0].shape[0])

    if min(cls_feats.shape[0],cls_feats.shape[1])>Nreps:
        kmeans = KMeans(Nreps).fit(cls_feats.transpose())
        c_cls_reps = kmeans.cluster_centers_.T
    else:
        logger.info('not enough samples for clustering')
        return [],[]
    c_cos_sim = np.dot(c_cls_reps.transpose(), cls_feats)  # dims=[#reps, #samples]
    c_coverage = np.max(c_cos_sim, axis=0)
    c_coverage_mean = np.mean(c_coverage)
    c_coverage_std = np.std(c_coverage)
    c_coverage_worst = np.min(c_coverage)
    c_coverage_which = np.argmax(c_cos_sim, axis=0)
    c_cover_size = np.zeros((Nreps))
    for i in range(Nreps):
        c_cover_size[i] = int(np.where(c_coverage_which==i)[0].shape[0])

    logger.info('trained: mean={0:.2f}, std={1:.2f}, worst={2:.2f}, min cluster:{3} max cluster:{4} '.format(coverage_mean,coverage_std,coverage_worst,np.min(cover_size), np.max(cover_size)))
    logger.info('cluster sizes: '+str(cover_size))
    logger.info('clustered: mean={0:.2f}, std={1:.2f}, worst={2:.2f}, min cluster:{3} max cluster:{4}'.format(c_coverage_mean, c_coverage_std, c_coverage_worst, np.min(c_cover_size),
                                                                                                      np.max(c_cover_size)))
    logger.info('cluster sizes: ' + str(c_cover_size))
    reps_stats = np.asarray([coverage_mean,coverage_std,coverage_worst, np.min(cover_size), np.max(cover_size),
                             c_coverage_mean,c_coverage_std,c_coverage_worst,np.min(c_cover_size), np.max(c_cover_size)])
    return reps_stats,c_cls_reps


class MutableModule(BaseModule):
    """A mutable module is a module that supports variable input data.

    Parameters
    ----------
    symbol : Symbol
    data_names : list of str
    label_names : list of str
    logger : Logger
    context : Context or list of Context
    work_load_list : list of number
    max_data_shapes : list of (name, shape) tuple, designating inputs whose shape vary
    max_label_shapes : list of (name, shape) tuple, designating inputs whose shape vary
    fixed_param_prefix : list of str, indicating fixed parameters
    """
    def __init__(self, symbol, data_names, label_names,
                 logger=logging, context=ctx.cpu(), work_load_list=None,
                 max_data_shapes=None, max_label_shapes=None, fixed_param_prefix=None, alt_fixed_param_prefix=None):
        super(MutableModule, self).__init__(logger=logger)
        self._symbol = symbol
        self._data_names = data_names
        self._label_names = label_names
        self._context = context
        self._work_load_list = work_load_list

        self._curr_module = None
        self._max_data_shapes = max_data_shapes
        self._max_label_shapes = max_label_shapes
        self._fixed_param_prefix = fixed_param_prefix
        self._alt_fixed_param_prefix = alt_fixed_param_prefix

        fixed_param_names = list()
        if fixed_param_prefix is not None:
            for name in self._symbol.list_arguments():
                for prefix in self._fixed_param_prefix:
                    if prefix in name:
                        fixed_param_names.append(name)
        self._fixed_param_names = fixed_param_names

        alt_fixed_param_names = list()
        if alt_fixed_param_prefix is not None:
            for name in self._symbol.list_arguments():
                for prefix in self._alt_fixed_param_prefix:
                    if prefix in name:
                        alt_fixed_param_names.append(name)
        self._alt_fixed_param_names = alt_fixed_param_names

        self._preload_opt_states = None

        self.config = None

    def _reset_bind(self):
        self.binded = False
        self._curr_module = None

    @property
    def data_names(self):
        return self._data_names

    @property
    def output_names(self):
        return self._symbol.list_outputs()

    @property
    def data_shapes(self):
        assert self.binded
        return self._curr_module.data_shapes

    @property
    def label_shapes(self):
        assert self.binded
        return self._curr_module.label_shapes

    @property
    def output_shapes(self):
        assert self.binded
        return self._curr_module.output_shapes

    def get_params(self):
        assert self.binded and self.params_initialized
        return self._curr_module.get_params()

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False, allow_extra=False):
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'
        self._curr_module.init_params(initializer=initializer, arg_params=arg_params,
                                      aux_params=aux_params, allow_missing=allow_missing,
                                      force_init=force_init)
        self.params_initialized = True

    def gather_samples(self,train_data,config):

        ovp_thresh = 0.7
        bg_thresh = 0.3
        bg_rois_per_image = 10
        num_ex_per_class = config.TRAIN.NUMEX_FOR_CLUSTERING

        cls_embeds = [[] for _ in range(10000)]
        cls_counts = np.zeros((10000,))
        max_cls = 0
        class_ords = []
        tic = time.time()
        bg_embeds = np.zeros((config.network.EMBEDDING_DIM, 0))
        for nbatch, data_batch in enumerate(train_data):
            print(nbatch)
            num_filled = np.sum(cls_counts >= num_ex_per_class)
            num_met = np.sum(cls_counts > 0)

            if (max_cls > 0) and (num_filled >= (config.dataset.NUM_CLASSES - 1)):
                break

            if nbatch % 100 == 0:
                toc = time.time()
                self.logger.info(
                    '[{0:.2f}] Computing embedding for batch #{1}, num. classes filled = {2}, max_cls = {3}, num met = {4}'.format(
                        toc - tic, nbatch, num_filled, max_cls, num_met))
                tic = time.time()

            # check batch, if all classes covered - skip
            to_skip = True
            for iB, bp in enumerate(data_batch.data):
                gt_boxes = bp[2][0].asnumpy()
                classes = np.array(gt_boxes[:, 4]).astype(int)
                if np.any(cls_counts[classes] < num_ex_per_class):
                    to_skip = False
                    break

            if to_skip:
                continue

            self.forward(data_batch, is_train=False)  # with train fails on CUDA out of memory

            for iB, bp in enumerate(data_batch.data):
                rois = self._curr_module._exec_group.execs[iB].outputs[-2].asnumpy()
                embed = self._curr_module._exec_group.execs[iB].outputs[-1].asnumpy()
                gt_boxes = bp[2][0].asnumpy()
                boxes = gt_boxes[:, 0:4]
                classes = gt_boxes[:, 4]
                ovp = bb_overlap(rois[:, 1:], gt_boxes)
                for iC, C in enumerate(classes):
                    class_ords.append([int(C)])
                    valid = ovp[:, iC] >= ovp_thresh
                    cls_embeds[int(C)].extend(embed[valid])
                    cls_counts[int(C)] += int(np.any(valid))
                    max_cls = np.maximum(max_cls, int(C))
                if config.network.BG_REPS:
                    bg_indices = np.where(np.max(ovp, axis=1) < bg_thresh)[0]
                    bg_indices = random.sample(bg_indices, min(len(bg_indices), bg_rois_per_image))
                    bg_embeds = np.concatenate((bg_embeds, embed[bg_indices].transpose()), axis=1)

        class_ords = np.unique(class_ords).tolist()
        return cls_embeds,bg_embeds,class_ords


    def run_vis_reps_TSNE(self, config,train_data, suffix,epoch,vis_samples=None,bg_samples=None):
        train_data.reset()
        arg_params = self._curr_module._arg_params
        reps = arg_params['fc_representatives_weight'].asnumpy()
        if config.network.BG_REPS:
            bg_reps = arg_params['fc_bg_reps_weight'].asnumpy()
        else:
            bg_reps = []
        print_root = assert_folder(os.path.join('/dccstor/jsdata1/dev/RepMet/output/dev/vis_reps/inst_{0}'.format(config.test_idx)))

        pars = edict()
        pars.Edim = config.network.EMBEDDING_DIM
        pars.dpi_value = 1200  # dpi of produced image
        pars.GroupSize = 8  # groups of classes printed separately for better visualization
        pars.REP_L2_NORM = config.network.REP_L2_NORM
        pars.do_BG = config.network.BG_REPS  # if background reps are also available
        pars.Nclasses = config.dataset.NUM_CLASSES - 1  # int(reps.shape[0] / (config.network.EMBEDDING_DIM * config.network.REPS_PER_CLASS))
        pars.REPS_PER_CLASS = config.network.REPS_PER_CLASS
        pars.vis_reps_fname_pref = os.path.join(print_root, 'vis_reps_' + suffix)
        pars.vis_bg_reps_fname = os.path.join(print_root, 'vis_bg_reps.jpg')

        reps = np.reshape(reps, (config.network.REPS_DIM, config.network.REPS_PER_CLASS, pars.Nclasses))
        if config.network.REP_L2_NORM:
            reps = normalize_reps(reps)
            if config.network.BG_REPS:
                bg_reps = normalize_reps(bg_reps)

        if config.network.BG_REPS:
            vis_bg_reps = np.reshape(bg_reps, (config.network.EMBEDDING_DIM, config.network.BG_REPS_NUMBER))
        else:
            vis_bg_reps = []


        # pars.X_embedded_fname = os.path.join(print_root, 'vis_reps_data.pkl')
        if vis_samples == None:
            cls_embeds, bg_embeds,class_ords = self.gather_samples(train_data,config)
            if len(class_ords) != pars.Nclasses:
                self.logger.info('Error: len(class_ords) != pars.Nclasses')
            vis_samples = []
            for ord in class_ords:
                vis_samples.append(np.array(cls_embeds[ord]).transpose())
            # vis_samples = []
            # for idx in range(pars.Nclasses):
            #     vis_samples.append(np.array(cls_embeds[idx + 1]).transpose())
        reps_stats = [ [] for _ in range(pars.Nclasses)]
        for idx,ord in enumerate(class_ords):
            cls_reps = reps[:,:,idx]
            cls_feats = np.array(cls_embeds[ord]).transpose()
            reps_stats[idx] = get_reps_stats(cls_reps,cls_feats)

        vis_reps_TSNE(vis_samples, reps, bg_embeds, vis_bg_reps, pars)
        train_data.reset()
        return vis_samples,bg_embeds

    def compute_rep_stats(self,train_data, config):
        train_data.reset()
        cls_embeds, bg_embeds, class_ords = self.gather_samples(train_data, config)
        if len(class_ords) != config.dataset.NUM_CLASSES - 1:
            self.logger.info('Error: numbers of classes doesn`t match')
            os.error('Error: numbers of classes doesn`t match')
        self.logger.info('Computing reps statistics ------------------------------------')
        print('Computing reps statistics ------------------------------------')
        arg_params = self._curr_module._arg_params
        reps = arg_params['fc_representatives_weight'].asnumpy()
        if config.network.BG_REPS:
            bg_reps = arg_params['fc_bg_reps_weight'].asnumpy()
        else:
            bg_reps = []
        if config.network.DEEP_REPS:
            reps_dim = 1024
        else:
            reps_dim = config.network.EMBEDDING_DIM
        reps = np.reshape(reps, (reps_dim, config.network.REPS_PER_CLASS, config.dataset.NUM_CLASSES - 1))
        if config.network.REP_L2_NORM:
            reps = normalize_reps(reps)
            if config.network.BG_REPS:
                bg_reps = normalize_reps(bg_reps)

        reps_stats_cls = np.zeros((0, 10))
        reps_kmeans = np.zeros(shape=reps.shape)
        for idx in range(config.dataset.NUM_CLASSES - 1):
            self.logger.info('-----------class {0} ------------'.format(idx))
            ord = class_ords[idx]
            cls_reps = reps[:, :, idx]
            cls_feats = np.array(cls_embeds[ord]).transpose()
            cls_stats, reps_kmeans[:,:,idx] = get_reps_stats(cls_reps, cls_feats, config,self.logger)
            reps_stats_cls = np.concatenate((reps_stats_cls, np.expand_dims(cls_stats, axis=0)), axis=0)
        rs = np.mean(reps_stats_cls, axis=0)

        self.logger.info('Repres: mean prox {0:.3f}, std prox {1:.3f}, smallest prox {2:.3f}, smallest cluster {3:.3f}, largest cluster {4:.3f}'
                         .format(rs[0],rs[1],rs[2],rs[3],rs[4]))
        self.logger.info('Kmeans: mean prox {0:.3f}, std prox {1:.3f}, smallest prox {2:.3f}, smallest cluster {3:.3f}, largest cluster {4:.3f}'
                         .format(rs[5], rs[6], rs[7], rs[8], rs[9]))
        train_data.reset()
        return reps_kmeans

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None, grad_req='write'):
        # in case we already initialized params, keep it
        if self.params_initialized:
            arg_params, aux_params = self.get_params()

        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        assert shared_module is None, 'shared_module for MutableModule is not supported'

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True

        max_shapes_dict = dict()
        if self._max_data_shapes is not None:
            max_shapes_dict.update(dict(self._max_data_shapes[0]))
        if self._max_label_shapes is not None:
            max_shapes_dict.update(dict(self._max_label_shapes[0]))

        max_data_shapes = list()
        for name, shape in data_shapes[0]:
            if name in max_shapes_dict:
                max_data_shapes.append((name, max_shapes_dict[name]))
            else:
                max_data_shapes.append((name, shape))

        max_label_shapes = list()
        if not label_shapes.count(None) == len(label_shapes):
            for name, shape in label_shapes[0]:
                if name in max_shapes_dict:
                    max_label_shapes.append((name, max_shapes_dict[name]))
                else:
                    max_label_shapes.append((name, shape))

        if len(max_label_shapes) == 0:
            max_label_shapes = None

        module = Module(self._symbol, self._data_names, self._label_names, logger=self.logger,
                        context=self._context, work_load_list=self._work_load_list,
                        fixed_param_names=self._fixed_param_names, alt_fixed_param_names=self._alt_fixed_param_names)
        module.bind([max_data_shapes for _ in range(len(self._context))], [max_label_shapes for _ in range(len(self._context))],
                    for_training, inputs_need_grad, force_rebind=False, shared_module=None)
        self._curr_module = module

        # copy back saved params, if already initialized
        if self.params_initialized:
            self.set_params(arg_params, aux_params)

    def save_checkpoint(self, prefix, epoch, save_optimizer_states=False):
        """Save current progress to checkpoint.
        Use mx.callback.module_checkpoint as epoch_end_callback to save during training.

        Parameters
        ----------
        prefix : str
            The file prefix to checkpoint to
        epoch : int
            The current epoch number
        save_optimizer_states : bool
            Whether to save optimizer states for continue training
        """
        self._curr_module.save_checkpoint(prefix, epoch, save_optimizer_states)

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        assert self.binded and self.params_initialized
        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring.')
            return

        self._curr_module._preload_opt_states = self._preload_opt_states
        self._curr_module.init_optimizer(kvstore, optimizer, optimizer_params,
                                         force_init=force_init)
        self.optimizer_initialized = True

    def update_reps_via_clustering(self, epoch, train_data, config = None):
        from bbox.bbox_utils import bb_overlap
        import numpy as np
        import os
        from scipy.spatial.distance import cdist
        if not config.network.DEEP_REPS:
            config.network.REPS_DIM = config.network.EMBEDDING_DIM
        path_to_embeds = os.path.join(config.final_output_path,'cls_embeds_epoch_{0}.pkl'.format(int(epoch)))
        if not os.path.isfile(path_to_embeds):
            cls_embeds, bg_embeds, class_ords = self.gather_samples(train_data,config)
            with open(path_to_embeds, 'wb') as fid:
                cPickle.dump({'cls_embeds':cls_embeds,'bg_embeds':bg_embeds,'class_ords':class_ords}, fid, protocol=cPickle.HIGHEST_PROTOCOL)
        else:
            with open(path_to_embeds, 'rb') as fid:
                sample_data = cPickle.load(fid)
                cls_embeds = sample_data['cls_embeds']
                bg_embeds = sample_data['bg_embeds']
                class_ords = sample_data['class_ords']

        # if len(class_ords)!=config.dataset.NUM_CLASSES-1:
        #     print('Error: numbers of classes doesn`t match')
        #     os.error('Error: numbers of classes doesn`t match')


        arg_params = self._curr_module._arg_params
        aux_params = self._curr_module._aux_params
        reps = arg_params['fc_representatives_weight'].asnumpy()
        if config.network.BG_REPS:
            bg_reps = arg_params['fc_bg_reps_weight'].asnumpy()

# #-- JS --------------------------------------------------------------------------------
#         reps = np.reshape(reps, (config.network.EMBEDDING_DIM, config.network.REPS_PER_CLASS, config.dataset.NUM_CLASSES-1))
#         if config.network.REP_L2_NORM:
#             reps = normalize_reps(reps)
#             if config.network.BG_REPS:
#                 bg_reps = normalize_reps(bg_reps)
#
#         reps_stats_cls = np.zeros((0,6))
#         for idx in range(config.dataset.NUM_CLASSES-1):
#             ord = class_ords[idx]
#             cls_reps = reps[:, :, idx]
#             cls_feats = np.array(cls_embeds[ord]).transpose()
#             reps_stats_cls = np.concatenate((reps_stats_cls,np.expand_dims(get_reps_stats(cls_reps, cls_feats, config),axis=0)),axis=0)
#         print(np.mean(reps_stats_cls,axis=0))
# # -- JS --------------------------------------------------------------------------------

        n_clusters = config.network.REPS_PER_CLASS
        tic = time.time()
        allClst = []
        for iCls in range(config.dataset.NUM_CLASSES-1): #range(max_cls):
            if iCls % 100 == 0:
                toc = time.time()
                print('[{0:.2f}] processing class #{1}'.format(toc - tic, iCls))
                tic = time.time()
            samples = np.array(cls_embeds[iCls + 1])
            if (samples.size > 0) and (samples.shape[0] >= n_clusters):
                kmeans = KMeans(n_clusters).fit(samples)
                clst = kmeans.cluster_centers_.T
            else:
                clst = np.zeros((config.network.REPS_DIM,n_clusters))
            allClst.append(clst)
        reps_mat = np.stack(allClst, axis=2)

        if config.network.BG_REPS:
            bg_reps = arg_params['fc_bg_reps_weight'].asnumpy()
        # desired shape shape=(cfg.network.EMBEDDING_DIM, cfg.network.REPS_PER_CLASS, max_cls)


        # TSNE-visualize samples and representatives -----------------------------
        if config.network.VISUALIZE_REPS:
            pars = edict()
            pars.Edim = config.network.EMBEDDING_DIM
            pars.dpi_value = 1200 # dpi of produced image
            pars.GroupSize = 8 # groups of classes printed separately for better visualization
            pars.REP_L2_NORM = config.network.REP_L2_NORM
            pars.do_BG = config.network.BG_REPS # if background reps are also available
            pars.Nclasses = config.dataset.NUM_CLASSES - 1# int(reps.shape[0] / (config.network.EMBEDDING_DIM * config.network.REPS_PER_CLASS))
            vis_reps = np.reshape(reps, (config.network.EMBEDDING_DIM, config.network.REPS_PER_CLASS, pars.Nclasses))
            if config.network.BG_REPS:
                vis_bg_reps = np.reshape(bg_reps, (config.network.EMBEDDING_DIM,config.network.BG_REPS_NUMBER))
            else:
                vis_bg_reps = []
            pars.REPS_PER_CLASS = config.network.REPS_PER_CLASS
            print_root = assert_folder(os.path.join('/dccstor/jsdata1/dev/RepMet/output/dev','epoch_{0}_trained'.format(epoch)))
            pars.vis_reps_fname_pref = os.path.join(print_root,'vis_reps')
            pars.vis_bg_reps_fname = os.path.join(print_root,'vis_bg_reps.jpg')
            pars.X_embedded_fname = os.path.join(print_root,'vis_reps_data.pkl')
            vis_samples = []
            for idx in range(pars.Nclasses):
                vis_samples.append(np.array(cls_embeds[idx + 1]).transpose())
            kmeans = KMeans(config.network.BG_REPS_NUMBER).fit(bg_samples.transpose())
            bg_reps_mat = kmeans.cluster_centers_.T

        if config.network.VISUALIZE_REPS:  # nbatch%10==0 and
            suffix = 'trained_e{0}_trained'.format(epoch)
            vis_samples, bg_samples = self.run_vis_reps_TSNE(config,train_data, suffix, epoch)

        # replace the reps with reps_mat --------------------------------
        reps = reps_mat.reshape(reps.shape)
        arg_params['fc_representatives_weight'] = nd.array(reps)
        if config.network.BG_REPS:
            bg_reps = bg_reps_mat.reshape(bg_reps.shape)
            arg_params['fc_bg_reps_weight'] = nd.array(bg_reps)
        self._curr_module.set_params(arg_params, aux_params)

        if config.network.store_kmeans_reps:
            self.save_checkpoint(config.network.model_kmeans_fname, epoch, save_optimizer_states=False)
            return

        if config.network.VISUALIZE_REPS:  # nbatch%10==0 and
            suffix = 'trained_e{0}_clustered'.format(epoch)
            self.run_vis_reps_TSNE(config,train_data, suffix, epoch,vis_samples,bg_samples)

        train_data.reset()


    # Leonid: debug code for storing problematic images with annotations
    def show_boxes_simple(self, im, boxes, classes, scale=1.0, save_file_path='temp.png'):
        import matplotlib.pyplot as plt
        from random import random as rand

        fig = plt.figure(1)
        fig.set_size_inches((2 * 8.5, 2 * 11), forward=False)

        plt.cla()
        plt.axis("off")
        plt.imshow(im)
        for iBox, box in enumerate(boxes):
            bbox = box[:4] * scale
            color = (rand(), rand(), rand())
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=2.5)
            plt.gca().add_patch(rect)

            if box.shape[0] == 5:
                cls_id = box[-1] - 1 # removing the BG class
                cls_name = classes[int(cls_id)]
                plt.gca().text(bbox[0], bbox[1],
                               '{:s}'.format(cls_name),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
        plt.show()
        fig.savefig(save_file_path)

        return

    import pickle
    # with open('pascal_imagenet_classes.pkl', 'r') as handle:
    #     classes = pickle.load(handle)

    def wrap_compute_rep_stats(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None, prefix=None, state=None, config=None,train_stop_event=None):

        self.config = config
        label_shapes = train_data.provide_label
        if config.network.base_net_lock:
            label_shapes = [None]*len(self._context)
        self.bind(data_shapes=train_data.provide_data, label_shapes=label_shapes,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)
        if state is not None:
            self._curr_module.load_optimizer_states(state)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        stop_training = False
        if config.network.compute_rep_stats:
            reps_kmeans = self.compute_rep_stats(train_data, config)


    def fit_resumable(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None, prefix=None, state=None, config=None,train_stop_event=None,is_resume=False):
        """Train the module parameters.

        Parameters
        ----------
        train_data : DataIter
        eval_data : DataIter
            If not `None`, will be used as validation set and evaluate the performance
            after each epoch.
        eval_metric : str or EvalMetric
            Default `'acc'`. The performance measure used to display during training.
        epoch_end_callback : function or list of function
            Each callback will be called with the current `epoch`, `symbol`, `arg_params`
            and `aux_params`.
        batch_end_callback : function or list of function
            Each callback will be called with a `BatchEndParam`.
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The parameters for the optimizer constructor.
            The default value is not a `dict`, just to avoid pylint warning on dangerous
            default values.
        eval_end_callback : function or list of function
            These will be called at the end of each full evaluation, with the metrics over
            the entire evaluation set.
        eval_batch_end_callback : function or list of function
            These will be called at the end of each minibatch during evaluation
        initializer : Initializer
            Will be called to initialize the module parameters if not already initialized.
        arg_params : dict
            Default `None`, if not `None`, should be existing parameters from a trained
            model or loaded from a checkpoint (previously saved model). In this case,
            the value here will be used to initialize the module parameters, unless they
            are already initialized by the user via a call to `init_params` or `fit`.
            `arg_params` has higher priority to `initializer`.
        aux_params : dict
            Default `None`. Similar to `arg_params`, except for auxiliary states.
        allow_missing : bool
            Default `False`. Indicate whether we allow missing parameters when `arg_params`
            and `aux_params` are not `None`. If this is `True`, then the missing parameters
            will be initialized via the `initializer`.
        force_rebind : bool
            Default `False`. Whether to force rebinding the executors if already binded.
        force_init : bool
            Default `False`. Indicate whether we should force initialization even if the
            parameters are already initialized.
        begin_epoch : int
            Default `0`. Indicate the starting epoch. Usually, if we are resuming from a
            checkpoint saved at a previous training phase at epoch N, then we should specify
            this value as N+1.
        num_epoch : int
            Number of epochs to run training.

        Examples
        --------
        An example of using fit for training::
            >>> #Assume training dataIter and validation dataIter are ready
            >>> mod.fit(train_data=train_dataiter, eval_data=val_dataiter,
                        optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
                        num_epoch=10)
        """
        assert num_epoch is not None, 'please specify number of epochs'

        self.config = config

        label_shapes = train_data.provide_label
        if config.network.base_net_lock:
            label_shapes = [None]*len(self._context)
        self.bind(data_shapes=train_data.provide_data, label_shapes=label_shapes,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if state is not None:
            self._curr_module.load_optimizer_states(state)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        stop_training = False
        if is_resume:
            begin_epoch-=1
        for epoch in range(begin_epoch, num_epoch):
            logging.debug("Fitting model. Epoch {}/{}".format(epoch, num_epoch))
            print('-------- training epoch {0} ---------------'.format(epoch+1))

            tic = time.time()
            eval_metric.reset()

            # K-means update before every second epoch (0,2,4..)
            if config.TRAIN.UPDATE_REPS_VIA_CLUSTERING:
                if epoch>=config.TRAIN.UPDATE_REPS_START_EPOCH and epoch<=config.TRAIN.UPDATE_REPS_STOP_EPOCH:
                    self.logger.info('----------- performing UPDATE_REPS_VIA_CLUSTERING -------------------')
                    print('----------- performing UPDATE_REPS_VIA_CLUSTERING -------------------')
                    self.update_reps_via_clustering(epoch, train_data, config)


            from core import callback

            for nbatch, data_batch in enumerate(train_data):
                #simulate train_stop_event:
                # if nbatch == 8:
                #     from threading import Event
                #     train_stop_event= Event()
                #     train_stop_event.set()



                a = data_batch.data[0][2][0].asnumpy()
                if any((a[:, 2] - a[:, 0])<=0):
                    logging.debug('negative width >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    continue
                if any((a[:, 3] - a[:, 1]) <=0):
                    logging.debug('negative height >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    continue
                if train_stop_event is not None and train_stop_event.is_set():
                    logging.debug("Stopping training.")
                    stop_training = True
                    break

                if monitor is not None:
                    monitor.tic()

                self.forward_backward(data_batch)

                self.update()

                self.update_metric(eval_metric, data_batch.label)

                if monitor is not None:
                    monitor.toc_print()

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)

            train_data.reset()

            if stop_training:
                # save intermediate data
                self.logger.info('Epoch[%d]: premature stop at batch %d', epoch, nbatch)
                arg_params, aux_params = self.get_params()

                # save epoch results
                if epoch_end_callback is not None:
                    for callback in _as_list(epoch_end_callback):
                        callback(epoch, self.symbol, arg_params, aux_params)

                from core import callback
                loc_end_callback = callback.Speedometer(train_data.batch_size, frequent=nbatch)
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                 eval_metric=eval_metric,
                                                 locals=locals())
                for callback in _as_list(loc_end_callback):
                    callback(batch_end_params)
                for callback in _as_list(loc_end_callback):
                    callback(batch_end_params)
                break


            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            # sync aux params across devices
            # if len(self._context)>1:
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)
            if JS_DEBUG:
                t = dictCompare(arg_params_b, arg_params)
                # reps = copy.deepcopy(arg_params['fc_representatives_weight'].asnumpy())

            if epoch_end_callback is not None:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            if eval_data: # evaluation on validation set
                res = self.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                #TODO: pull this into default
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

        if config.network.compute_rep_stats:
            reps_kmeans = self.compute_rep_stats(train_data, config)

            # end of 1 epoch, reset the data-iter for another epoch



        status = not stop_training
        return status,nbatch,train_stop_event


    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None, prefix=None, state=None, config=None,train_stop_event=None):
        """Train the module parameters.

        Parameters
        ----------
        train_data : DataIter
        eval_data : DataIter
            If not `None`, will be used as validation set and evaluate the performance
            after each epoch.
        eval_metric : str or EvalMetric
            Default `'acc'`. The performance measure used to display during training.
        epoch_end_callback : function or list of function
            Each callback will be called with the current `epoch`, `symbol`, `arg_params`
            and `aux_params`.
        batch_end_callback : function or list of function
            Each callback will be called with a `BatchEndParam`.
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The parameters for the optimizer constructor.
            The default value is not a `dict`, just to avoid pylint warning on dangerous
            default values.
        eval_end_callback : function or list of function
            These will be called at the end of each full evaluation, with the metrics over
            the entire evaluation set.
        eval_batch_end_callback : function or list of function
            These will be called at the end of each minibatch during evaluation
        initializer : Initializer
            Will be called to initialize the module parameters if not already initialized.
        arg_params : dict
            Default `None`, if not `None`, should be existing parameters from a trained
            model or loaded from a checkpoint (previously saved model). In this case,
            the value here will be used to initialize the module parameters, unless they
            are already initialized by the user via a call to `init_params` or `fit`.
            `arg_params` has higher priority to `initializer`.
        aux_params : dict
            Default `None`. Similar to `arg_params`, except for auxiliary states.
        allow_missing : bool
            Default `False`. Indicate whether we allow missing parameters when `arg_params`
            and `aux_params` are not `None`. If this is `True`, then the missing parameters
            will be initialized via the `initializer`.
        force_rebind : bool
            Default `False`. Whether to force rebinding the executors if already binded.
        force_init : bool
            Default `False`. Indicate whether we should force initialization even if the
            parameters are already initialized.
        begin_epoch : int
            Default `0`. Indicate the starting epoch. Usually, if we are resuming from a
            checkpoint saved at a previous training phase at epoch N, then we should specify
            this value as N+1.
        num_epoch : int
            Number of epochs to run training.

        Examples
        --------
        An example of using fit for training::
            >>> #Assume training dataIter and validation dataIter are ready
            >>> mod.fit(train_data=train_dataiter, eval_data=val_dataiter,
                        optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
                        num_epoch=10)
        """
        assert num_epoch is not None, 'please specify number of epochs'

        self.config = config

        label_shapes = train_data.provide_label
        if config.network.base_net_lock:
            label_shapes = [None]*len(self._context)
        self.bind(data_shapes=train_data.provide_data, label_shapes=label_shapes,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)
        if state is not None:
            self._curr_module.load_optimizer_states(state)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        stop_training = False

        # print metrics state before training --------------------------------------------
        if False:
            self.logger.info('print metrics state before training ----------------------')
            eval_metric.reset()
            for nbatch, data_batch in enumerate(train_data):
                self.forward_backward(data_batch)
                self.update_metric(eval_metric, data_batch.label)
            train_data.reset()
            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch -1', name, val)




        for epoch in range(begin_epoch, num_epoch):
            logging.info("Fitting model. Epoch {}/{}".format(epoch, num_epoch))
            print('-------- training epoch {0} ---------------'.format(epoch))

            tic = time.time()
            eval_metric.reset()
            if JS_DEBUG:
                arg_params, aux_params = self.get_params()
                arg_params_b = copy.deepcopy(arg_params)

            # K-means update before every second epoch (0,2,4..)
            #if (config.TRAIN.UPDATE_REPS_VIA_CLUSTERING and epoch%2==0) or epoch==config.TRAIN.begin_epoch or config.network.store_kmeans_reps:

            if config.TRAIN.UPDATE_REPS_VIA_CLUSTERING:
                if epoch>=config.TRAIN.UPDATE_REPS_START_EPOCH and epoch<=config.TRAIN.UPDATE_REPS_STOP_EPOCH:
                    self.logger.info('----------- performing UPDATE_REPS_VIA_CLUSTERING -------------------')
                    print('----------- performing UPDATE_REPS_VIA_CLUSTERING -------------------')
                    self.update_reps_via_clustering(epoch, train_data, config)

            if LEONID_PROFILING_ENABLED:
                tic1 = time.time()


            for nbatch, data_batch in enumerate(train_data):
                # if nbatch<3260:
                a = data_batch.data[0][2][0].asnumpy()
                # if nbatch%100==0:
                #     print('nbatch {0}'.format(nbatch))
                if any((a[:, 2] - a[:, 0])<=0):
                    logging.debug('negative width >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    continue
                if any((a[:, 3] - a[:, 1]) <=0):
                    logging.debug('negative height >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    continue
                if train_stop_event is not None and train_stop_event.is_set():
                    logging.debug("Stopping training.")
                    stop_training = True
                    break

                if monitor is not None:
                    monitor.tic()

                self.forward_backward(data_batch)

                self.update()

                self.update_metric(eval_metric, data_batch.label)

                if monitor is not None:
                    monitor.toc_print()

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)

            train_data.reset()

            if stop_training:
                break

            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            # sync aux params across devices
            # if len(self._context)>1:
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)
            if JS_DEBUG:
                t = dictCompare(arg_params_b, arg_params)
                # reps = copy.deepcopy(arg_params['fc_representatives_weight'].asnumpy())

            if epoch_end_callback is not None:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            if eval_data: # evaluation on validation set
                res = self.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                #TODO: pull this into default
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

        if config.network.compute_rep_stats:
            reps_kmeans = self.compute_rep_stats(train_data, config)

            # end of 1 epoch, reset the data-iter for another epoch



        status = not stop_training
        return status

    def fit_precompute(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None, prefix=None, state=None, config=None,train_stop_event=None, work_root=None,pre_idx=1):

        assert num_epoch is not None, 'please specify number of epochs'
        self.config = config

        label_shapes = train_data.provide_label
        if config.network.base_net_lock:
            label_shapes = [None]*len(self._context)
        self.bind(data_shapes=train_data.provide_data, label_shapes=label_shapes,
                  for_training=True, force_rebind=force_rebind)

        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)


        ################################################################################
        # precomputing loop
        ################################################################################


        for nbatch, data_batch in enumerate(train_data):
            _, im_fname = os.path.split(train_data.img_fname[0])
            rps_fname = os.path.join(work_root, 'precomputed_data', im_fname.replace('.jpg', '_feat.pkl'))
            if os.path.exists(rps_fname):
                continue
            if train_stop_event is not None and train_stop_event.is_set():
                logging.debug("Stopping training.")
                stop_training = True
                break

            a = data_batch.data[0][2][0].asnumpy()
            if any((a[:, 2] - a[:, 0])<=0):
                logging.debug('negative width >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                #continue
            if any((a[:, 3] - a[:, 1]) <=0):
                logging.debug('negative height >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                #continue

            self.forward_backward(data_batch)

            for iB, bp in enumerate(data_batch.data):
                if pre_idx==1:
                    bbox_weight_pre = self._curr_module._exec_group.execs[iB].outputs[-3].asnumpy()
                    bbox_target_pre = self._curr_module._exec_group.execs[iB].outputs[-4].asnumpy()
                    label_pre = self._curr_module._exec_group.execs[iB].outputs[-5].asnumpy()
                    fc_new_1_relu_pre = self._curr_module._exec_group.execs[iB].outputs[-6].asnumpy()
                    rois = self._curr_module._exec_group.execs[iB].outputs[-2].asnumpy()
                    state_data = {'fc_new_1_relu_pre': fc_new_1_relu_pre, 'label_pre': label_pre, 'bbox_target_pre': bbox_target_pre, 'bbox_weight_pre': bbox_weight_pre,'rois_pre':rois}

                _, im_fname = os.path.split(train_data.img_fname[0])
                rps_fname = os.path.join(work_root, 'precomputed_data', im_fname.replace('.jpg', '_feat.pkl'))
                with open(rps_fname, 'wb') as fid:
                    cPickle.dump(state_data,fid, protocol=cPickle.HIGHEST_PROTOCOL)

        train_data.reset()

        return None


    def fit_using_precompute(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None, prefix=None, state=None, config=None,train_stop_event=None):
        """Train the module parameters.

        Parameters
        ----------
        train_data : DataIter
        eval_data : DataIter
            If not `None`, will be used as validation set and evaluate the performance
            after each epoch.
        eval_metric : str or EvalMetric
            Default `'acc'`. The performance measure used to display during training.
        epoch_end_callback : function or list of function
            Each callback will be called with the current `epoch`, `symbol`, `arg_params`
            and `aux_params`.
        batch_end_callback : function or list of function
            Each callback will be called with a `BatchEndParam`.
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The parameters for the optimizer constructor.
            The default value is not a `dict`, just to avoid pylint warning on dangerous
            default values.
        eval_end_callback : function or list of function
            These will be called at the end of each full evaluation, with the metrics over
            the entire evaluation set.
        eval_batch_end_callback : function or list of function
            These will be called at the end of each minibatch during evaluation
        initializer : Initializer
            Will be called to initialize the module parameters if not already initialized.
        arg_params : dict
            Default `None`, if not `None`, should be existing parameters from a trained
            model or loaded from a checkpoint (previously saved model). In this case,
            the value here will be used to initialize the module parameters, unless they
            are already initialized by the user via a call to `init_params` or `fit`.
            `arg_params` has higher priority to `initializer`.
        aux_params : dict
            Default `None`. Similar to `arg_params`, except for auxiliary states.
        allow_missing : bool
            Default `False`. Indicate whether we allow missing parameters when `arg_params`
            and `aux_params` are not `None`. If this is `True`, then the missing parameters
            will be initialized via the `initializer`.
        force_rebind : bool
            Default `False`. Whether to force rebinding the executors if already binded.
        force_init : bool
            Default `False`. Indicate whether we should force initialization even if the
            parameters are already initialized.
        begin_epoch : int
            Default `0`. Indicate the starting epoch. Usually, if we are resuming from a
            checkpoint saved at a previous training phase at epoch N, then we should specify
            this value as N+1.
        num_epoch : int
            Number of epochs to run training.

        Examples
        --------
        An example of using fit for training::
            >>> #Assume training dataIter and validation dataIter are ready
            >>> mod.fit(train_data=train_dataiter, eval_data=val_dataiter,
                        optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
                        num_epoch=10)
        """
        assert num_epoch is not None, 'please specify number of epochs'
        assert num_epoch>0, 'please specify a positive number of epochs'

        self.config = config

        label_shapes = train_data.provide_label
        if config.network.base_net_lock:
            label_shapes = [None]*len(self._context)
        self.bind(data_shapes=train_data.provide_data, label_shapes=label_shapes,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)
        if state is not None:
            self._curr_module.load_optimizer_states(state)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        stop_training = False

        # print metrics state before training --------------------------------------------
        if False:
            self.logger.info('print metrics state before training ----------------------')
            eval_metric.reset()
            for nbatch, data_batch in enumerate(train_data):
                self.forward_backward(data_batch)
                self.update_metric(eval_metric, data_batch.label)
            train_data.reset()
            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch -1', name, val)

        #sw = SummaryWriter(logdir='./logs')################### mxboard

        for epoch in range(begin_epoch, num_epoch):
            logging.debug("Fitting model. Epoch {}/{}".format(epoch, num_epoch))
            print('-------- training epoch {0} ---------------'.format(epoch))

            tic = time.time()
            eval_metric.reset()
            if JS_DEBUG:
                arg_params, aux_params = self.get_params()
                arg_params_b = copy.deepcopy(arg_params)

            # K-means update before every second epoch (0,2,4..)
            #if (config.TRAIN.UPDATE_REPS_VIA_CLUSTERING and epoch%2==0) or epoch==config.TRAIN.begin_epoch or config.network.store_kmeans_reps:

            if config.TRAIN.UPDATE_REPS_VIA_CLUSTERING:
                if epoch>=config.TRAIN.UPDATE_REPS_START_EPOCH and epoch<=config.TRAIN.UPDATE_REPS_STOP_EPOCH:
                    self.logger.info('----------- performing UPDATE_REPS_VIA_CLUSTERING -------------------')
                    print('----------- performing UPDATE_REPS_VIA_CLUSTERING -------------------')
                    self.update_reps_via_clustering(epoch, train_data, config)

            if LEONID_PROFILING_ENABLED:
                tic1 = time.time()




            for nbatch, data_batch in enumerate(train_data):

                if monitor is not None:
                    monitor.tic()

                self.forward_backward(data_batch)

                LEONID_DEBUG = False
                if LEONID_DEBUG:
                    import numpy as np
                    from bbox.bbox_utils import bb_overlap
                    ovp_thresh = 0.7
                    num_TP = 0
                    num_tot = 0
                    num_in_loss = 0
                    num_in_loss_tot = 0
                    max_score_TP = 0
                    all_classes = []
                    for iB, bp in enumerate(data_batch.data):
                        if not config.network.base_net_lock:
                            rois = self._curr_module._exec_group.execs[iB].outputs[-2].asnumpy()
                            probs = self._curr_module._exec_group.execs[iB].outputs[2].asnumpy()
                            ohem_labels = self._curr_module._exec_group.execs[iB].outputs[4].asnumpy()
                        else:
                            rois = self._curr_module._exec_group.execs[iB].outputs[-2].asnumpy()
                            probs = self._curr_module._exec_group.execs[iB].outputs[0].asnumpy()
                            ohem_labels = self._curr_module._exec_group.execs[iB].outputs[2].asnumpy()
                        gt_boxes = bp[2][0].asnumpy()
                        boxes = gt_boxes[:, 0:4]
                        classes = gt_boxes[:, 4]
                        all_classes.append(classes)
                        ovp = bb_overlap(rois[:, 1:], gt_boxes)
                        for iC, C in enumerate(classes):
                            valid = ovp[:, iC] >= ovp_thresh
                            num_TP+=np.sum(np.argmax(probs[0,valid,:],axis=1)==int(C))
                            max_score_TP=np.maximum(max_score_TP,np.max(probs[0,valid,int(C)]))
                            num_tot+=np.sum(valid)
                            num_in_loss+=np.sum(ohem_labels[0,valid]>0)
                            num_in_loss_tot+=np.sum(ohem_labels>0)
                    all_classes=np.unique(np.concatenate(all_classes))
                    if (num_TP==0) and (epoch>=config.TRAIN.STORE_DEBUG_IMAGES_EPOCH):
                        # draw and store the problematic image with annotations
                        from utils.image import transform_inverse
                        import os
                        for iB, bp in enumerate(data_batch.data):
                            img=transform_inverse(bp[0].asnumpy(),config.network.PIXEL_MEANS)
                            gt_boxes = bp[2][0].asnumpy()
                            path_to_debug_images = os.path.join(config.final_output_path,
                                                          'debug_images','epoch_{0}'.format(int(epoch)))
                            if not os.path.exists(path_to_debug_images):
                                os.makedirs(path_to_debug_images)
                            path_to_img = os.path.join(path_to_debug_images,'batch_{0}_img_{1}.jpg'.format(nbatch,iB))
                            self.show_boxes_simple(img, gt_boxes, self.classes, scale=1.0, save_file_path=path_to_img)
                    print('TP={0},max_score_TP={5:.4f},tot={1},TP_in_loss={2},tot_in_loss={3},classes={4}'.format(num_TP,num_tot,num_in_loss,num_in_loss_tot,all_classes,max_score_TP))

                if LEONID_PROFILING_ENABLED:
                    toc1 = time.time()
                    print('self.forward_backward(data_batch) {0}'.format(toc1 - tic1))
                    tic1 = time.time()

                self.update()

                if LEONID_PROFILING_ENABLED:
                    toc1 = time.time()
                    print('self.update() {0}'.format(toc1 - tic1))
                    tic1 = time.time()

                self.update_metric(eval_metric, data_batch.label)

                if LEONID_PROFILING_ENABLED:
                    toc1 = time.time()
                    print('self.update_metric(eval_metric, data_batch.label) {0}'.format(toc1 - tic1))
                    tic1 = time.time()

                if monitor is not None:
                    monitor.toc_print()

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)

                if LEONID_PROFILING_ENABLED:
                    toc1 = time.time()
                    print('batch_end_callback {0}'.format(toc1 - tic1))
                    tic1 = time.time()

            train_data.reset()

            if stop_training:
                break

            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
                #sw.add_scalar(tag=name, value=val, global_step=epoch)################### mxboard
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            # sync aux params across devices
            # if len(self._context)>1:
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)
            if JS_DEBUG:
                t = dictCompare(arg_params_b, arg_params)
                # reps = copy.deepcopy(arg_params['fc_representatives_weight'].asnumpy())

            if epoch_end_callback is not None:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            if eval_data: # evaluation on validation set
                res = self.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                #TODO: pull this into default
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

        # if config.network.compute_rep_stats:
        #     reps_kmeans = self.compute_rep_stats(train_data, config)

            # end of 1 epoch, reset the data-iter for another epoch



        status = not stop_training
        return status



    def forward(self, data_batch, is_train=None):
        assert self.binded and self.params_initialized

        if LEONID_PROFILING_ENABLED:
            tic = time.time()

        # get current_shapes
        if self._curr_module.label_shapes is not None:
            current_shapes = [dict(self._curr_module.data_shapes[i] + self._curr_module.label_shapes[i]) for i in range(len(self._context))]
        else:
            current_shapes = [dict(self._curr_module.data_shapes[i]) for i in range(len(self._context))]

        if LEONID_PROFILING_ENABLED:
            toc = time.time()
            print('MutableModule::forward 1 {0}'.format(toc - tic))
            tic = time.time()

        # get input_shapes
        if is_train and (not self.config.network.base_net_lock):
            input_shapes = [dict(data_batch.provide_data[i] + data_batch.provide_label[i]) for i in range(len(self._context))]
        else:
            input_shapes = [dict(data_batch.provide_data[i]) for i in range(len(data_batch.provide_data))]

        if LEONID_PROFILING_ENABLED:
            toc = time.time()
            print('MutableModule::forward 2 {0}'.format(toc - tic))
            tic = time.time()

        # decide if shape changed
        shape_changed = len(current_shapes[0]) != len(input_shapes[0])
        if not shape_changed:
            for pre, cur in zip(current_shapes, input_shapes):
                for k, v in pre.items():
                    if v != cur[k]:
                        shape_changed = True

        if (self.config is not None )and self.config.network.FORCE_RESHAPE_EVERY_BATCH:
            shape_changed = True

        if LEONID_PROFILING_ENABLED:
            toc = time.time()
            print('MutableModule::forward 3 {0}'.format(toc - tic))
            tic = time.time()

        if shape_changed:
            if LEONID_PROFILING_ENABLED:
                tic1 = time.time()

            # self._curr_module.reshape(data_batch.provide_data, data_batch.provide_label)
            module = Module(self._symbol, self._data_names, self._label_names,
                            logger=self.logger, context=[self._context[i] for i in range(len(data_batch.provide_data))],
                            work_load_list=self._work_load_list,
                            fixed_param_names=self._fixed_param_names,
                            alt_fixed_param_names=self._alt_fixed_param_names)

            if LEONID_PROFILING_ENABLED:
                toc1 = time.time()
                print('MutableModule::forward 4a {0}'.format(toc1 - tic1))
                tic1 = time.time()

            labels = data_batch.provide_label
            if (self.config is not None) and self.config.network.base_net_lock:
                labels = [None]*len(self._context)
            module.bind(data_batch.provide_data, labels, self._curr_module.for_training,
                        self._curr_module.inputs_need_grad, force_rebind=False,
                        shared_module=self._curr_module)
            self._curr_module = module

            if LEONID_PROFILING_ENABLED:
                toc1 = time.time()
                print('MutableModule::forward 4b {0}'.format(toc1 - tic1))

        if LEONID_PROFILING_ENABLED:
            toc = time.time()
            print('MutableModule::forward 4 {0}'.format(toc - tic))
            tic = time.time()

        self._curr_module.forward(data_batch, is_train=is_train)

        if LEONID_PROFILING_ENABLED:
            toc = time.time()
            print('MutableModule::forward 5 {0}'.format(toc - tic))

    def backward(self, out_grads=None):
        assert self.binded and self.params_initialized
        self._curr_module.backward(out_grads=out_grads)

    def update(self):
        assert self.binded and self.params_initialized and self.optimizer_initialized
        self._curr_module.update()

    def get_outputs(self, merge_multi_context=True):
        assert self.binded and self.params_initialized
        return self._curr_module.get_outputs(merge_multi_context=merge_multi_context)
    def get_input_grads(self, merge_multi_context=True):
        assert self.binded and self.params_initialized and self.inputs_need_grad
        return self._curr_module.get_input_grads(merge_multi_context=merge_multi_context)

    def update_metric(self, eval_metric, labels):
        assert self.binded and self.params_initialized
        self._curr_module.update_metric(eval_metric, labels)

    def install_monitor(self, mon):
        """ Install monitor on all executors """
        assert self.binded
        self._curr_module.install_monitor(mon)
