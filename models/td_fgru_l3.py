#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.recurrent import mult_h_td_fgru as hgru


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        # Add input
        in_emb = conv.skinny_input_layer(
            X=data_tensor,
            reuse=reuse,
            training=training,
            features=16,
            conv_kernel_size=7,
            pool=False,
            name='l0')
        layer_hgru = hgru.hGRU(
            'fgru',
            x_shape=in_emb.get_shape().as_list(),
            timesteps=8,
            h_ext=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],  # hgru1, hgru2, fgru2, fgru1  # Increase h_ext ~ 15
            strides=[1, 1, 1, 1],
            hgru_ids={'h1': 16, 'h2': 64, 'h3': 128, 'fb2': 64, 'fb1': 16},  # hGRU labels and channels
            hgru_idx={'h1': 0, 'h2': 1, 'h3': 2, 'fb2': 3, 'fb1': 4},
            padding='SAME',
            aux={
                'readout': 'fb',
                'intermediate_ff': [64, 128],  # Add one more layer w/ either 36 or 72 channels
                'intermediate_ks': [[7, 7], [5, 5]],
                'while_loop': False,
                'skip': True,
                'include_pooling': True
            },
            pool_strides=[4, 4],
            pooling_kernel=[4, 4],
            train=training)
        h2 = layer_hgru.build(in_emb)
        h2 = normalization.batch(
            bottom=h2,
            renorm=False,
            name='hgru_bn',
            training=training)
        activity = conv.readout_layer(
            activity=h2,
            reuse=reuse,
            training=training,
            output_shape=output_shape)
    extra_activities = {
        'activity': h2
    }
    return activity, extra_activities
