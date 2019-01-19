"""Model training with tfrecord queues or placeholders."""
import os
import time
import tensorflow as tf
import numpy as np
from datetime import datetime
from utils import logger
from utils import py_utils
# from ops import data_to_tfrecords
from tqdm import tqdm
# from db import db


def val_status(
        log,
        dt,
        step,
        train_loss,
        rate,
        timer,
        score_function,
        train_score,
        val_score,
        summary_dir):
    """Print training status."""
    format_str = (
        '%s: step %d, loss = %.2f (%.1f examples/sec; '
        '%.3f sec/batch) | Training %s = %s | '
        'Validation %s = %s | logdir = %s')
    log.info(
        format_str % (
            dt,
            step,
            train_loss,
            rate,
            timer,
            score_function,
            train_score,
            score_function,
            val_score,
            summary_dir))


def train_status(
        log,
        dt,
        step,
        train_loss,
        rate,
        timer,
        score_function,
        train_score):
    """Print training status."""
    format_str = (
        '%s: step %d, loss = %.5f (%.1f examples/sec; '
        '%.3f sec/batch) | Training %s = %s')
    log.info(
        format_str % (
            dt,
            step,
            train_loss,
            rate,
            timer,
            score_function,
            train_score))


def training_step(
        sess,
        train_dict,
        feed_dict=False):
    """Run a step of training."""
    start_time = time.time()
    if feed_dict:
        it_train_dict = sess.run(train_dict, feed_dict=feed_dict)
    else:
        it_train_dict = sess.run(train_dict)
    train_score = it_train_dict['train_score']
    train_loss = it_train_dict['train_loss']
    duration = time.time() - start_time
    timesteps = duration
    return train_score, train_loss, it_train_dict, timesteps


def validation_step(
        sess,
        val_dict,
        config,
        log,
        val_images=False,
        val_labels=False,
        val_batch_idx=False,
        val_batches=False):
    it_val_score = np.asarray([])
    it_val_loss = np.asarray([])
    start_time = time.time()
    if val_batch_idx:
        shuff_val_batch_idx = val_batch_idx[
            np.random.permutation(len(val_batch_idx))]
    for num_vals in range(config.validation_steps):
        # Validation accuracy as the average of n batches
        if val_images:
            it_idx = shuff_val_batch_idx == num_vals
            it_ims = val_images[it_idx]
            it_labs = val_labels[it_idx]
            if isinstance(it_labs[0], basestring):
                it_labs = np.asarray(
                    [
                        data_to_tfrecords.load_image(im)
                        for im in it_labs])
            feed_dict = {
                val_dict['val_images']: it_ims,
                val_dict['val_labels']: it_labs
            }
            it_val_dict = sess.run(val_dict, feed_dict=feed_dict)
        else:
            it_val_dict = sess.run(val_dict)
        it_val_score = np.append(
            it_val_score,
            it_val_dict['val_score'])
        it_val_loss = np.append(
            it_val_loss,
            it_val_dict['val_loss'])
    val_score = it_val_score.mean()
    val_lo = it_val_loss.mean()
    duration = time.time() - start_time
    return val_score, val_lo, it_val_dict, duration


def save_progress(
        config,
        val_check,
        weight_dict,
        it_val_dict,
        exp_label,
        step,
        directories,
        sess,
        saver,
        val_score,
        val_loss,
        train_score,
        train_loss,
        timer,
        num_params,
        log,
        summary_op,
        summary_writer,
        save_activities,
        save_gradients,
        save_checkpoints):
    """Save progress and important data."""
    if config.save_weights and val_check:
        it_weights = {
            k: it_val_dict[k] for k in weight_dict.keys()}
        py_utils.save_npys(
            data=it_weights,
            model_name='%s_%s' % (
                exp_label,
                step),
            output_string=directories['weights'])

    if save_activities and val_check:
        py_utils.save_npys(
            data=it_val_dict,
            model_name='%s_%s' % (
                exp_label,
                step),
            output_string=directories['weights'])

    ckpt_path = os.path.join(
        directories['checkpoints'],
        'model_%s.ckpt' % step)
    if save_checkpoints and val_check:
        saver.save(
            sess,
            ckpt_path,
            global_step=step)

    if save_gradients and val_check:
        # np.savez(
        #     os.path.join(
        #         config.results,
        #         '%s_train_gradients' % exp_label),
        #     **it_train_dict)
        np.savez(
            os.path.join(
                config.results,
                '%s_val_gradients' % exp_label),
            **it_val_dict)
    db.update_performance(
        experiment_id=config._id,
        experiment=config.experiment,
        train_score=float(train_score),
        train_loss=float(train_loss),
        val_score=float(val_score),
        val_loss=float(val_loss),
        step=step,
        num_params=int(num_params),
        ckpt_path=ckpt_path,
        results_path=config.results,
        summary_path=directories['summaries'])

    # Summaries
    summary_str = sess.run(summary_op)
    summary_writer.add_summary(summary_str, step)


def training_loop(
        config,
        coord,
        sess,
        summary_op,
        summary_writer,
        saver,
        threads,
        directories,
        train_dict,
        val_dict,
        exp_label,
        num_params,
        placeholders=False,
        checkpoint=None,
        save_weights=False,
        save_checkpoints=False,
        save_activities=False,
        save_gradients=False):
    """Run the model training loop."""
    log = logger.get(os.path.join(config.log_dir, exp_label))
    val_perf = np.asarray([np.inf])
    step = 0
    if save_weights:
        try:
            weight_dict = {v.name: v for v in tf.trainable_variables()}
            val_dict = dict(
                val_dict,
                **weight_dict)
        except Exception:
            raise RuntimeError('Failed to find weights to save.')
    else:
        weight_dict = None
    if placeholders:
        placeholder_images = placeholders[0]
        placeholder_labels = placeholders[1]
        train_images = placeholder_images['train']
        val_images = placeholder_images['val']
        train_labels = placeholder_labels['train']
        val_labels = placeholder_labels['val']
        train_batches = len(train_images) / config.train_batch_size
        train_batch_idx = np.arange(
            train_batches / config.train_batch_size).reshape(-1, 1).repeat(
                config.train_batch_size)
        train_images = train_images[:len(train_batch_idx)]
        train_labels = train_labels[:len(train_batch_idx)]
        val_batches = len(val_images) / config.val_batch_size
        val_batch_idx = np.arange(
            val_batches / config.val_batch_size).reshape(-1, 1).repeat(
                config.val_batch_size)
        val_images = val_images[:len(val_batch_idx)]
        val_labels = val_labels[:len(val_batch_idx)]
        raise NotImplementedError
        for epoch in tqdm(
                range(config.epochs),
                desc='Epoch',
                total=config.epochs):
            for train_batch in range(train_batches):
                data_idx = train_batch_idx == train_batch
                it_train_images = train_images[data_idx]
                it_train_labels = train_labels[data_idx]
                if isinstance(it_train_images[0], basestring):
                    it_train_images = np.asarray(
                        [
                            data_to_tfrecords.load_image(im)
                            for im in it_train_images])
                feed_dict = {
                    train_dict['train_images']: it_train_images,
                    train_dict['train_labels']: it_train_labels
                }
                (
                    train_score,
                    train_loss,
                    it_train_dict,
                    timer) = training_step(
                    sess=sess,
                    train_dict=train_dict,
                    feed_dict=feed_dict)
                if step % config.validation_period == 0:
                    val_score, val_lo, it_val_dict, duration = validation_step(
                        sess=sess,
                        val_dict=val_dict,
                        config=config,
                        log=log,
                        val_images=val_images,
                        val_labels=val_labels,
                        val_batch_idx=val_batch_idx,
                        val_batches=val_batches)

                    # Save progress and important data
                    try:
                        val_check = np.where(val_lo < val_perf)[0]
                        save_progress(
                            config=config,
                            val_check=val_check,
                            weight_dict=weight_dict,
                            it_val_dict=it_val_dict,
                            exp_label=exp_label,
                            step=step,
                            directories=directories,
                            sess=sess,
                            saver=saver,
                            val_score=val_score,
                            val_loss=val_lo,
                            train_score=train_score,
                            train_loss=train_loss,
                            timer=duration,
                            num_params=num_params,
                            log=log,
                            summary_op=summary_op,
                            summary_writer=summary_writer,
                            save_activities=save_activities,
                            save_gradients=save_gradients,
                            save_checkpoints=save_checkpoints)
                    except Exception as e:
                        log.info('Failed to save checkpoint: %s' % e)

                    # Training status and validation accuracy
                    val_status(
                        log=log,
                        dt=datetime.now(),
                        step=step,
                        train_loss=train_loss,
                        rate=config.val_batch_size / duration,
                        timer=float(duration),
                        score_function=config.score_function,
                        train_score=train_score,
                        val_score=val_score,
                        summary_dir=directories['summaries'])
                    val_perf = np.concatenate([val_perf, [val_lo]])
                else:
                    # Training status
                    train_status(
                        log=log,
                        dt=datetime.now(),
                        step=step,
                        train_loss=train_loss,
                        rate=config.val_batch_size / duration,
                        timer=float(duration),
                        score_function=config.score_function,
                        train_score=train_score)
                # End iteration
                step += 1

    else:
        try:
            while not coord.should_stop():
                (
                    train_score,
                    train_loss,
                    it_train_dict,
                    duration) = training_step(
                    sess=sess,
                    train_dict=train_dict)
                if step % config.validation_period == 0:
                    val_score, val_lo, it_val_dict, duration = validation_step(
                        sess=sess,
                        val_dict=val_dict,
                        config=config,
                        log=log)

                    # Save progress and important data
                    try:
                        val_check = np.where(val_lo < val_perf)[0]
                        save_progress(
                            config=config,
                            val_check=val_check,
                            weight_dict=weight_dict,
                            it_val_dict=it_val_dict,
                            exp_label=exp_label,
                            step=step,
                            directories=directories,
                            sess=sess,
                            saver=saver,
                            val_score=val_score,
                            val_loss=val_lo,
                            train_score=train_score,
                            train_loss=train_loss,
                            timer=duration,
                            num_params=num_params,
                            log=log,
                            summary_op=summary_op,
                            summary_writer=summary_writer,
                            save_activities=save_activities,
                            save_gradients=save_gradients,
                            save_checkpoints=save_checkpoints)
                    except Exception as e:
                        log.info('Failed to save checkpoint: %s' % e)

                    # Training status and validation accuracy
                    val_status(
                        log=log,
                        dt=datetime.now(),
                        step=step,
                        train_loss=train_loss,
                        rate=config.val_batch_size / duration,
                        timer=float(duration),
                        score_function=config.score_function,
                        train_score=train_score,
                        val_score=val_score,
                        summary_dir=directories['summaries'])
                else:
                    # Training status
                    train_status(
                        log=log,
                        dt=datetime.now(),
                        step=step,
                        train_loss=train_loss,
                        rate=config.val_batch_size / duration,
                        timer=float(duration),
                        score_function=config.score_function,
                        train_score=train_score)

                # End iteration
                val_perf = np.concatenate([val_perf, [val_lo]])
                step += 1
        except tf.errors.OutOfRangeError:
            log.info(
                'Done training for %d epochs, %d steps.' % (
                    config.epochs, step))
            log.info('Saved to: %s' % directories['checkpoints'])
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
    return

