# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import argparse
import os
sys.path.append('VideoProcessing')
import VideoProcessor

import numpy as np
import tensorflow as tf

from config import*

import i3d

_IMAGE_SIZE = 224
_NUM_CLASSES = 400

_SAMPLE_VIDEO_FRAMES = 79


_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}
_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}
 
_LABEL_MAP_PATH = 'data/label_map.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')


def main(unused_argv):

  # parser = argparse.ArgumentParser(description='Process some integers.')
  # parser.add_argument('video_clip_num', metavar='N', type=int, help='an integer for the accumulator')

  # args = parser.parse_args().video_clip_num

  # i = 1
  # with open("training_lists/test_list.txt", "r") as fin:
  #   for path in fin.readlines():
  #     if(i < args):
  #       i+=1
  #       continue
  #     temp = "testData/" + path.split("/")[1]
  #     print(temp)
  #     if(os.path.exists("testData/" + path.split("/")[1].strip())):
  #       print("hello")
  #       _SAMPLE_PATHS['rgb']  = "testData/" + path.split("/")[1].strip() + "/out_rgb.npy"
  #       _SAMPLE_PATHS['flow']  = "testData/" + path.split("/")[1].strip() + "/out_flow.npy"
  #       break



  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type
  imagenet_pretrained = FLAGS.imagenet_pretrained

  # Has to be RGB/Flow/Joint
  if eval_type not in ['rgb', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, flow, joint')

  # kinetics classes are mapped
  kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

  if eval_type in ['rgb', 'joint']:
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)
    rgb_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'RGB':
        rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

  if eval_type in ['flow', 'joint']:
    # Flow input has only 2 channels.
    flow_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
    with tf.variable_scope('Flow'):
      flow_model = i3d.InceptionI3d(
          _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      flow_logits, _ = flow_model(
          flow_input, is_training=False, dropout_keep_prob=1.0)
    flow_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

  if eval_type == 'rgb':
    model_logits = rgb_logits
  elif eval_type == 'flow':
    model_logits = flow_logits
  else:
    model_logits = rgb_logits + flow_logits
  model_predictions = tf.nn.softmax(model_logits)









  #Where the training actually takes place
  with tf.Session() as sess:
    feed_dict = {}
    if eval_type in ['rgb', 'joint']:
      print(imagenet_pretrained)
      if imagenet_pretrained:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
        # rgb_saver.restore(sess,MAYBEWORK_DIR)
        # flow_saver.restore(sess,MAYBEWORK_DIR)
        #z = {**rgb_variable_map, **flow_variable_map}
        print("Printing Keys:")
        # for key in z:
        #   print(key)
        # saver = tf.train.Saver(var_list=z)
        # sess.run(tf.global_variables_initializer())
        # saver.restore(sess,MAYBEWORK_DIR)
      else:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
      tf.logging.info('RGB checkpoint restored')
      rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
      tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
      feed_dict[rgb_input] = rgb_sample

    if eval_type in ['flow', 'joint']:
      if imagenet_pretrained:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
        print("Do nothing, already loaded")
      else:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
      tf.logging.info('Flow checkpoint restored')
      flow_sample = np.load(_SAMPLE_PATHS['flow'])### Where the test data gets loaded in
      tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
      feed_dict[flow_input] = flow_sample

    out_logits, out_predictions = sess.run(
        [model_logits, model_predictions],
        feed_dict=feed_dict)

    out_logits = out_logits[0]
    out_predictions = out_predictions[0]
    sorted_indices = np.argsort(out_predictions)[::-1]

    print('Norm of logits: %f' % np.linalg.norm(out_logits))
    print('\nTop classes and probabilities')
    print(eval_type)

    # with open("testResults.txt", "+a") as fout:
    #   fout.write("Ground truth class clip # " + str(args) + ": " + _SAMPLE_PATHS["rgb"].split("_")[1] + "\n")
    #   with open("topLabelResult.txt", "a") as fout2:
    #     fout2.write("Ground truth: " +  '{0: <18}'.format(_SAMPLE_PATHS["rgb"].split("_")[1]) + " Predicted: " +  kinetics_classes[sorted_indices[0]] + "\n")
    #   for index in sorted_indices[:20]:
    #     print(out_predictions[index], out_logits[index], kinetics_classes[index])
    #     fout.write(str(out_predictions[index]) + "   " + str(out_logits[index]) + "   " + str(kinetics_classes[index]) + "\n")
    #   fout.write("\n")
    for index in sorted_indices[:20]:
      print(out_predictions[index], out_logits[index], kinetics_classes[index])

if __name__ == '__main__':
  main("")




