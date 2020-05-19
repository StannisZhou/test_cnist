# import numpy as np
import tensorflow as tf

# from ops import metrics
# from ops import data_structure
from ops import data_loader, gradients, losses, optimizers, tf_fun, training
from utils import py_utils


def get_placeholders(dataset_module, config):
    """Create placeholders and apply augmentations."""
    raise NotImplementedError
    train_images = tf.placeholder(
        dtype=dataset_module.tf_reader['image']['dtype'],
        shape=[config.batch_size] + dataset_module.im_size,
        name='train_images',
    )
    train_labels = tf.placeholder(
        dtype=dataset_module.tf_reader['label']['dtype'],
        shape=[config.batch_size] + dataset_module.label_size,
        name='train_labels',
    )
    val_images = tf.placeholder(
        dtype=dataset_module.tf_reader['image']['dtype'],
        shape=[config.batch_size] + dataset_module.im_size,
        name='val_images',
    )
    val_labels = tf.placeholder(
        dtype=dataset_module.tf_reader['label']['dtype'],
        shape=[config.batch_size] + dataset_module.label_size,
        name='val_labels',
    )
    aug_train_ims, aug_train_labels = [], []
    aug_val_ims, aug_val_labels = [], []
    split_train_ims = tf.split(train_images, config.batch_size, axis=0)
    split_train_labels = tf.split(train_labels, config.batch_size, axis=0)
    split_val_ims = tf.split(val_images, config.batch_size, axis=0)
    split_val_labels = tf.split(val_labels, config.batch_size, axis=0)
    for tr_im, tr_la, va_im, va_la in zip(
        split_train_ims, split_train_labels, split_val_ims, split_val_labels
    ):
        tr_im, tr_la = data_loader.image_augmentations(
            image=tf.squeeze(tr_im),
            label=tf.squeeze(tr_la),
            model_input_image_size=dataset_module.model_input_image_size,
            data_augmentations=config.data_augmentations,
        )
        va_im, va_la = data_loader.image_augmentations(
            image=tf.squeeze(va_im),
            label=tf.squeeze(va_la),
            model_input_image_size=dataset_module.model_input_image_size,
            data_augmentations=config.val_augmentations,
        )
        aug_train_ims += [tr_im]
        aug_train_labels += [tr_la]
        aug_val_ims += [va_im]
        aug_val_labels += [va_la]
    aug_train_ims = tf.stack(aug_train_ims, axis=0)
    aug_train_labels = tf.stack(aug_train_labels, axis=0)
    aug_val_ims = tf.stack(aug_val_ims, axis=0)
    aug_val_labels = tf.stack(aug_val_labels, axis=0)
    return aug_train_ims, aug_train_labels, aug_val_ims, aug_val_labels


def build_model(
    exp_params,
    config,
    log,
    dt_string,
    gpu_device,
    cpu_device,
    placeholders=False,
    checkpoint=None,
    tensorboard_images=False,
):
    """Standard model building routines."""
    config = py_utils.add_to_config(d=exp_params, config=config)
    exp_label = '%s_%s' % (exp_params['experiment'], py_utils.get_dt_stamp())
    directories = py_utils.prepare_directories(config, exp_label)
    dataset_module = py_utils.import_module(
        pre_path=config.dataset_classes, module=config.train_dataset
    )
    train_dataset_module = dataset_module.data_processing()
    (train_data, train_means_image, train_means_label) = py_utils.get_data_pointers(
        dataset=train_dataset_module.output_name, base_dir=config.tf_records, cv='train'
    )
    dataset_module = py_utils.import_module(
        pre_path=config.dataset_classes, module=config.val_dataset
    )
    val_dataset_module = dataset_module.data_processing()
    val_data, val_means_image, val_means_label = py_utils.get_data_pointers(
        dataset=val_dataset_module.output_name, base_dir=config.tf_records, cv='val'
    )

    # Create data tensors
    if hasattr(train_dataset_module, 'aux_loss'):
        train_aux_loss = train_dataset_module.aux_loss
    else:
        train_aux_loss = None
    with tf.device(cpu_device):
        if placeholders:
            raise NotImplementedError
            (train_images, train_labels, val_images, val_labels) = get_placeholders(
                dataset_module, config
            )
            placeholders = dataset_module.get_data()
        else:
            train_images, train_labels, train_aux = data_loader.inputs(
                dataset=train_data,
                batch_size=config.train_batch_size,
                model_input_image_size=train_dataset_module.model_input_image_size,
                tf_dict=train_dataset_module.tf_dict,
                data_augmentations=config.train_augmentations,
                num_epochs=config.epochs,
                aux=train_aux_loss,
                tf_reader_settings=train_dataset_module.tf_reader,
                shuffle=config.shuffle_train,
            )
            val_images, val_labels, val_aux = data_loader.inputs(
                dataset=val_data,
                batch_size=config.val_batch_size,
                model_input_image_size=val_dataset_module.model_input_image_size,
                tf_dict=val_dataset_module.tf_dict,
                data_augmentations=config.val_augmentations,
                num_epochs=None,
                tf_reader_settings=val_dataset_module.tf_reader,
                shuffle=config.shuffle_val,
            )

    # Build training and val models
    model_spec = py_utils.import_module(
        module=config.model, pre_path=config.model_classes
    )
    with tf.device(gpu_device):
        train_logits, train_vars = model_spec.build_model(
            data_tensor=train_images,
            reuse=None,
            training=True,
            output_shape=train_dataset_module.output_size,
        )
        val_logits, val_vars = model_spec.build_model(
            data_tensor=val_images,
            reuse=tf.AUTO_REUSE,
            training=False,
            output_shape=val_dataset_module.output_size,
        )

    # Derive loss
    train_loss = losses.derive_loss(
        labels=train_labels, logits=train_logits, loss_type=config.loss_function
    )
    val_loss = losses.derive_loss(
        labels=val_labels, logits=val_logits, loss_type=config.loss_function
    )
    tf.summary.scalar('train_loss', train_loss)
    tf.summary.scalar('val_loss', val_loss)

    # Derive auxilary losses
    if hasattr(train_dataset_module, 'aux_loss'):
        for k, v in train_vars.iteritems():
            if k in train_dataset_module.aux_loss.keys():
                aux_loss_type, scale = train_dataset_module.aux_loss[k]
                train_loss += (
                    losses.derive_loss(
                        labels=train_aux, logits=v, loss_type=aux_loss_type
                    )
                    * scale
                )

    # Derive score
    train_score = losses.derive_score(
        labels=train_labels,
        logits=train_logits,
        loss_type=config.loss_function,
        score_type=config.score_function,
    )
    val_score = losses.derive_score(
        labels=val_labels,
        logits=val_logits,
        loss_type=config.loss_function,
        score_type=config.score_function,
    )
    tf.summary.scalar('train_score', train_score)
    tf.summary.scalar('val_score', val_score)
    if tensorboard_images:
        tf.summary.image('train_images', train_images)
        tf.summary.image('val_images', val_images)

    # Build optimizer
    train_op = optimizers.get_optimizer(train_loss, config.lr, config.optimizer)

    # Initialize tf variables
    saver = tf.train.Saver(
        var_list=tf.global_variables(), max_to_keep=config.save_checkpoints
    )
    summary_op = tf.summary.merge_all()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(
        tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    )
    summary_writer = tf.summary.FileWriter(directories['summaries'], sess.graph)
    if not placeholders:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    else:
        coord, threads = None, None

    # Create dictionaries of important training and validation information
    train_dict = {
        'train_loss': train_loss,
        'train_score': train_score,
        'train_images': train_images,
        'train_labels': train_labels,
        'train_logits': train_logits,
        'train_op': train_op,
        'train_aux': train_aux,
    }
    if isinstance(train_vars, dict):
        for k, v in train_vars.iteritems():
            train_dict[k] = v
    else:
        train_dict['activity'] = train_vars
    if hasattr(config, 'save_gradients'):
        grad = tf.gradients(train_logits, train_images)[0]
        if grad is not None:
            train_dict['gradients'] = grad
        else:
            log.warning('Could not calculate val gradients.')

    val_dict = {
        'val_loss': val_loss,
        'val_score': val_score,
        'val_images': val_images,
        'val_logits': val_logits,
        'val_labels': val_labels,
        'val_aux': val_aux,
    }
    if isinstance(val_vars, dict):
        for k, v in val_vars.iteritems():
            val_dict[k] = v
    else:
        val_dict['activity'] = val_vars
    if hasattr(config, 'save_gradients'):
        grad = tf.gradients(val_logits, val_images)[0]
        if grad is not None:
            val_dict['gradients'] = grad
        else:
            log.warning('Could not calculate val gradients.')

    # Count parameters
    num_params = tf_fun.count_parameters(var_list=tf.trainable_variables())
    print 'Model has approximately %s trainable params.' % num_params

    # Start training loop
    training.training_loop(
        config=config,
        coord=coord,
        sess=sess,
        summary_op=summary_op,
        summary_writer=summary_writer,
        saver=saver,
        threads=threads,
        directories=directories,
        train_dict=train_dict,
        val_dict=val_dict,
        exp_label=exp_label,
        num_params=num_params,
        checkpoint=checkpoint,
        save_weights=config.save_weights,
        save_checkpoints=config.save_checkpoints,
        save_activities=config.save_activities,
        save_gradients=config.save_gradients,
        placeholders=placeholders,
    )
