# @title All `dm_control` imports required for this tutorial
from distutils.command.config import config

# The basic mujoco wrapper.
from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

# Control Suite
from dm_control import suite

# Run through corridor example
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks

# Soccer
from dm_control.locomotion import soccer

# Manipulation
from dm_control import manipulation
# @title Other imports and helper functions

# General
import copy
import os
import time
import itertools
from IPython.display import clear_output
import numpy as np

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from IPython.display import HTML
import PIL.Image
import matplotlib.animation as manimation

# Internal loading of video libraries.

# Use svg backend for figure rendering
# %config InlineBackend.figure_format = 'svg'

# Font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['animation.ffmpeg_path'] = '/Users/jiyunhyo/Downloads/ffmpeg'

# Inline video helper function
if os.environ.get('COLAB_NOTEBOOK_TEST', False):
    # We skip video generation during tests, as it is quite expensive.
    display_video = lambda *args, **kwargs: None
else:
    def display_video(frames, framerate, video_name):
        height, width, _ = frames[0].shape
        dpi = 1000
        orig_backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
        fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
        matplotlib.use(orig_backend)  # Switch back to the original backend.
        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.set_position([0, 0, 1, 1])
        im = ax.imshow(frames[0])

        def update(frame):
            im.set_data(frame)
            return [im]

        interval = 1000 / framerate
        anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                       interval=interval, blit=True, repeat=False)
        anim.save(video_name+ "_" + str(dpi)+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
        return anim

# Seed numpy's global RNG so that cell outputs are deterministic. We also try to
# use RandomState instances that are local to a single cell wherever possible.
np.random.seed(42)

# @title A static model {vertical-output: true}

static_model = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""
physics = mujoco.Physics.from_xml_string(static_model)
pixels = physics.render()
okay = PIL.Image.fromarray(pixels)
# okay.show()

# @title A child body with a joint { vertical-output: true }

swinging_body = """
<!-- Copyright 2021 DeepMind Technologies Limited

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

         http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
-->

<mujoco model="22 Humanoids">
  <option timestep="0.005"/>

  <size nconmax="1000" njmax="3000"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="body" type="cube" builtin="flat" mark="cross" width="127" height="1278"
             rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>
    <material name="body" texture="body" texuniform="true" rgba="0.8 0.6 .4 1"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>

  <default>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
    <default class="body">
      <geom type="capsule" condim="1" friction=".7" solimp=".9 .99 .003" solref=".015 1" material="body"/>
      <joint type="hinge" damping=".2" stiffness="1" armature=".01" limited="true" solimplimit="0 .99 .01"/>
      <default class="big_joint">
        <joint damping="5" stiffness="10"/>
        <default class="big_stiff_joint">
          <joint stiffness="20"/>
        </default>
      </default>
    </default>
  </default>

  <visual>
    <map force="0.1" zfar="30"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="4096"/>
    <global offwidth="800" offheight="800"/>
  </visual>

  <worldbody>
    <geom size="10 10 .05" type="plane" material="grid" condim="3"/>
    <light dir=".2 1 -.4" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="-2 -10 4" cutoff="35"/>
    <light dir="-.2 1 -.4" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="2 -10 4" cutoff="35"/>

    <body name="1a_torso" pos="-1 0 1.5" childclass="body">
      <camera name="1a_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="1a_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="1a_root"/>
      <geom name="1a_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="1a_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="1a_head" pos="0 0 .19">
        <geom name="1a_head" type="sphere" size=".09"/>
        <camera name="1a_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="1a_lower_waist" pos="-.01 0 -.26">
        <geom name="1a_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="1a_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="1a_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="1a_pelvis" pos="0 0 -.165">
          <joint name="1a_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="1a_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="1a_right_thigh" pos="0 -.1 -.04">
            <joint name="1a_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="1a_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="1a_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="1a_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="1a_right_shin" pos="0 .01 -.403">
              <joint name="1a_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="1a_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="1a_right_foot" pos="0 0 -.39">
                <joint name="1a_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="1a_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="1a_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="1a_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="1a_left_thigh" pos="0 .1 -.04">
            <joint name="1a_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="1a_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="1a_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="1a_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="1a_left_shin" pos="0 -.01 -.403">
              <joint name="1a_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="1a_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="1a_left_foot" pos="0 0 -.39">
                <joint name="1a_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="1a_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="1a_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="1a_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="1a_right_upper_arm" pos="0 -.17 .06">
        <joint name="1a_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="1a_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="1a_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="1a_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="1a_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="1a_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="1a_right_hand" pos=".18 .18 .18">
            <geom name="1a_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="1a_left_upper_arm" pos="0 .17 .06">
        <joint name="1a_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="1a_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="1a_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="1a_left_lower_arm" pos=".18 .18 -.18">
          <joint name="1a_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="1a_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="1a_left_hand" pos=".18 -.18 .18">
            <geom name="1a_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="2a_torso" pos="-1 1 1.5" childclass="body">
      <camera name="2a_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="2a_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="2a_root"/>
      <geom name="2a_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="2a_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="2a_head" pos="0 0 .19">
        <geom name="2a_head" type="sphere" size=".09"/>
        <camera name="2a_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="2a_lower_waist" pos="-.01 0 -.26">
        <geom name="2a_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="2a_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="2a_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="2a_pelvis" pos="0 0 -.165">
          <joint name="2a_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="2a_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="2a_right_thigh" pos="0 -.1 -.04">
            <joint name="2a_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="2a_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="2a_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="2a_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="2a_right_shin" pos="0 .01 -.403">
              <joint name="2a_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="2a_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="2a_right_foot" pos="0 0 -.39">
                <joint name="2a_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="2a_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="2a_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="2a_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="2a_left_thigh" pos="0 .1 -.04">
            <joint name="2a_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="2a_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="2a_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="2a_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="2a_left_shin" pos="0 -.01 -.403">
              <joint name="2a_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="2a_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="2a_left_foot" pos="0 0 -.39">
                <joint name="2a_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="2a_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="2a_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="2a_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="2a_right_upper_arm" pos="0 -.17 .06">
        <joint name="2a_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="2a_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="2a_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="2a_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="2a_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="2a_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="2a_right_hand" pos=".18 .18 .18">
            <geom name="2a_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="2a_left_upper_arm" pos="0 .17 .06">
        <joint name="2a_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="2a_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="2a_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="2a_left_lower_arm" pos=".18 .18 -.18">
          <joint name="2a_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="2a_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="2a_left_hand" pos=".18 -.18 .18">
            <geom name="2a_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="3a_torso" pos="-1 -1 1.5" childclass="body">
      <camera name="3a_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="3a_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="3a_root"/>
      <geom name="3a_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="3a_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="3a_head" pos="0 0 .19">
        <geom name="3a_head" type="sphere" size=".09"/>
        <camera name="3a_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="3a_lower_waist" pos="-.01 0 -.26">
        <geom name="3a_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="3a_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="3a_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="3a_pelvis" pos="0 0 -.165">
          <joint name="3a_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="3a_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="3a_right_thigh" pos="0 -.1 -.04">
            <joint name="3a_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="3a_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="3a_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="3a_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="3a_right_shin" pos="0 .01 -.403">
              <joint name="3a_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="3a_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="3a_right_foot" pos="0 0 -.39">
                <joint name="3a_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="3a_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="3a_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="3a_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="3a_left_thigh" pos="0 .1 -.04">
            <joint name="3a_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="3a_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="3a_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="3a_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="3a_left_shin" pos="0 -.01 -.403">
              <joint name="3a_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="3a_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="3a_left_foot" pos="0 0 -.39">
                <joint name="3a_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="3a_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="3a_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="3a_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="3a_right_upper_arm" pos="0 -.17 .06">
        <joint name="3a_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="3a_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="3a_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="3a_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="3a_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="3a_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="3a_right_hand" pos=".18 .18 .18">
            <geom name="3a_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="3a_left_upper_arm" pos="0 .17 .06">
        <joint name="3a_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="3a_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="3a_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="3a_left_lower_arm" pos=".18 .18 -.18">
          <joint name="3a_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="3a_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="3a_left_hand" pos=".18 -.18 .18">
            <geom name="3a_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="4a_torso" pos="-1 2 1.5" childclass="body">
      <camera name="4a_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="4a_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="4a_root"/>
      <geom name="4a_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="4a_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="4a_head" pos="0 0 .19">
        <geom name="4a_head" type="sphere" size=".09"/>
        <camera name="4a_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="4a_lower_waist" pos="-.01 0 -.26">
        <geom name="4a_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="4a_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="4a_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="4a_pelvis" pos="0 0 -.165">
          <joint name="4a_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="4a_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="4a_right_thigh" pos="0 -.1 -.04">
            <joint name="4a_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="4a_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="4a_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="4a_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="4a_right_shin" pos="0 .01 -.403">
              <joint name="4a_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="4a_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="4a_right_foot" pos="0 0 -.39">
                <joint name="4a_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="4a_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="4a_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="4a_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="4a_left_thigh" pos="0 .1 -.04">
            <joint name="4a_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="4a_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="4a_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="4a_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="4a_left_shin" pos="0 -.01 -.403">
              <joint name="4a_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="4a_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="4a_left_foot" pos="0 0 -.39">
                <joint name="4a_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="4a_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="4a_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="4a_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="4a_right_upper_arm" pos="0 -.17 .06">
        <joint name="4a_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="4a_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="4a_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="4a_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="4a_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="4a_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="4a_right_hand" pos=".18 .18 .18">
            <geom name="4a_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="4a_left_upper_arm" pos="0 .17 .06">
        <joint name="4a_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="4a_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="4a_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="4a_left_lower_arm" pos=".18 .18 -.18">
          <joint name="4a_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="4a_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="4a_left_hand" pos=".18 -.18 .18">
            <geom name="4a_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="5a_torso" pos="-1 -2 1.5" childclass="body">
      <camera name="5a_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="5a_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="5a_root"/>
      <geom name="5a_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="5a_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="5a_head" pos="0 0 .19">
        <geom name="5a_head" type="sphere" size=".09"/>
        <camera name="5a_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="5a_lower_waist" pos="-.01 0 -.26">
        <geom name="5a_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="5a_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="5a_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="5a_pelvis" pos="0 0 -.165">
          <joint name="5a_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="5a_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="5a_right_thigh" pos="0 -.1 -.04">
            <joint name="5a_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="5a_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="5a_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="5a_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="5a_right_shin" pos="0 .01 -.403">
              <joint name="5a_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="5a_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="5a_right_foot" pos="0 0 -.39">
                <joint name="5a_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="5a_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="5a_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="5a_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="5a_left_thigh" pos="0 .1 -.04">
            <joint name="5a_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="5a_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="5a_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="5a_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="5a_left_shin" pos="0 -.01 -.403">
              <joint name="5a_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="5a_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="5a_left_foot" pos="0 0 -.39">
                <joint name="5a_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="5a_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="5a_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="5a_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="5a_right_upper_arm" pos="0 -.17 .06">
        <joint name="5a_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="5a_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="5a_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="5a_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="5a_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="5a_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="5a_right_hand" pos=".18 .18 .18">
            <geom name="5a_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="5a_left_upper_arm" pos="0 .17 .06">
        <joint name="5a_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="5a_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="5a_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="5a_left_lower_arm" pos=".18 .18 -.18">
          <joint name="5a_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="5a_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="5a_left_hand" pos=".18 -.18 .18">
            <geom name="5a_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="6a_torso" pos="-1 3 1.5" childclass="body">
      <camera name="6a_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="6a_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="6a_root"/>
      <geom name="6a_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="6a_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="6a_head" pos="0 0 .19">
        <geom name="6a_head" type="sphere" size=".09"/>
        <camera name="6a_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="6a_lower_waist" pos="-.01 0 -.26">
        <geom name="6a_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="6a_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="6a_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="6a_pelvis" pos="0 0 -.165">
          <joint name="6a_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="6a_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="6a_right_thigh" pos="0 -.1 -.04">
            <joint name="6a_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="6a_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="6a_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="6a_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="6a_right_shin" pos="0 .01 -.403">
              <joint name="6a_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="6a_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="6a_right_foot" pos="0 0 -.39">
                <joint name="6a_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="6a_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="6a_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="6a_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="6a_left_thigh" pos="0 .1 -.04">
            <joint name="6a_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="6a_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="6a_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="6a_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="6a_left_shin" pos="0 -.01 -.403">
              <joint name="6a_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="6a_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="6a_left_foot" pos="0 0 -.39">
                <joint name="6a_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="6a_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="6a_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="6a_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="6a_right_upper_arm" pos="0 -.17 .06">
        <joint name="6a_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="6a_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="6a_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="6a_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="6a_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="6a_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="6a_right_hand" pos=".18 .18 .18">
            <geom name="6a_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="6a_left_upper_arm" pos="0 .17 .06">
        <joint name="6a_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="6a_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="6a_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="6a_left_lower_arm" pos=".18 .18 -.18">
          <joint name="6a_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="6a_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="6a_left_hand" pos=".18 -.18 .18">
            <geom name="6a_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="7a_torso" pos="-1 -3 1.5" childclass="body">
      <camera name="7a_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="7a_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="7a_root"/>
      <geom name="7a_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="7a_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="7a_head" pos="0 0 .19">
        <geom name="7a_head" type="sphere" size=".09"/>
        <camera name="7a_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="7a_lower_waist" pos="-.01 0 -.26">
        <geom name="7a_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="7a_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="7a_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="7a_pelvis" pos="0 0 -.165">
          <joint name="7a_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="7a_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="7a_right_thigh" pos="0 -.1 -.04">
            <joint name="7a_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="7a_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="7a_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="7a_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="7a_right_shin" pos="0 .01 -.403">
              <joint name="7a_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="7a_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="7a_right_foot" pos="0 0 -.39">
                <joint name="7a_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="7a_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="7a_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="7a_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="7a_left_thigh" pos="0 .1 -.04">
            <joint name="7a_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="7a_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="7a_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="7a_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="7a_left_shin" pos="0 -.01 -.403">
              <joint name="7a_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="7a_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="7a_left_foot" pos="0 0 -.39">
                <joint name="7a_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="7a_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="7a_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="7a_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="7a_right_upper_arm" pos="0 -.17 .06">
        <joint name="7a_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="7a_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="7a_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="7a_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="7a_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="7a_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="7a_right_hand" pos=".18 .18 .18">
            <geom name="7a_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="7a_left_upper_arm" pos="0 .17 .06">
        <joint name="7a_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="7a_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="7a_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="7a_left_lower_arm" pos=".18 .18 -.18">
          <joint name="7a_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="7a_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="7a_left_hand" pos=".18 -.18 .18">
            <geom name="7a_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="8a_torso" pos="-1 4 1.5" childclass="body">
      <camera name="8a_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="8a_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="8a_root"/>
      <geom name="8a_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="8a_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="8a_head" pos="0 0 .19">
        <geom name="8a_head" type="sphere" size=".09"/>
        <camera name="8a_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="8a_lower_waist" pos="-.01 0 -.26">
        <geom name="8a_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="8a_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="8a_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="8a_pelvis" pos="0 0 -.165">
          <joint name="8a_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="8a_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="8a_right_thigh" pos="0 -.1 -.04">
            <joint name="8a_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="8a_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="8a_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="8a_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="8a_right_shin" pos="0 .01 -.403">
              <joint name="8a_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="8a_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="8a_right_foot" pos="0 0 -.39">
                <joint name="8a_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="8a_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="8a_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="8a_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="8a_left_thigh" pos="0 .1 -.04">
            <joint name="8a_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="8a_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="8a_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="8a_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="8a_left_shin" pos="0 -.01 -.403">
              <joint name="8a_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="8a_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="8a_left_foot" pos="0 0 -.39">
                <joint name="8a_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="8a_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="8a_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="8a_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="8a_right_upper_arm" pos="0 -.17 .06">
        <joint name="8a_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="8a_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="8a_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="8a_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="8a_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="8a_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="8a_right_hand" pos=".18 .18 .18">
            <geom name="8a_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="8a_left_upper_arm" pos="0 .17 .06">
        <joint name="8a_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="8a_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="8a_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="8a_left_lower_arm" pos=".18 .18 -.18">
          <joint name="8a_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="8a_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="8a_left_hand" pos=".18 -.18 .18">
            <geom name="8a_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="9a_torso" pos="-1 -4 1.5" childclass="body">
      <camera name="9a_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="9a_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="9a_root"/>
      <geom name="9a_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="9a_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="9a_head" pos="0 0 .19">
        <geom name="9a_head" type="sphere" size=".09"/>
        <camera name="9a_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="9a_lower_waist" pos="-.01 0 -.26">
        <geom name="9a_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="9a_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="9a_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="9a_pelvis" pos="0 0 -.165">
          <joint name="9a_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="9a_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="9a_right_thigh" pos="0 -.1 -.04">
            <joint name="9a_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="9a_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="9a_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="9a_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="9a_right_shin" pos="0 .01 -.403">
              <joint name="9a_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="9a_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="9a_right_foot" pos="0 0 -.39">
                <joint name="9a_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="9a_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="9a_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="9a_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="9a_left_thigh" pos="0 .1 -.04">
            <joint name="9a_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="9a_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="9a_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="9a_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="9a_left_shin" pos="0 -.01 -.403">
              <joint name="9a_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="9a_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="9a_left_foot" pos="0 0 -.39">
                <joint name="9a_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="9a_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="9a_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="9a_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="9a_right_upper_arm" pos="0 -.17 .06">
        <joint name="9a_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="9a_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="9a_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="9a_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="9a_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="9a_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="9a_right_hand" pos=".18 .18 .18">
            <geom name="9a_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="9a_left_upper_arm" pos="0 .17 .06">
        <joint name="9a_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="9a_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="9a_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="9a_left_lower_arm" pos=".18 .18 -.18">
          <joint name="9a_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="9a_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="9a_left_hand" pos=".18 -.18 .18">
            <geom name="9a_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="10a_torso" pos="-1 5 1.5" childclass="body">
      <camera name="10a_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="10a_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="10a_root"/>
      <geom name="10a_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="10a_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="10a_head" pos="0 0 .19">
        <geom name="10a_head" type="sphere" size=".09"/>
        <camera name="10a_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="10a_lower_waist" pos="-.01 0 -.26">
        <geom name="10a_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="10a_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="10a_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="10a_pelvis" pos="0 0 -.165">
          <joint name="10a_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="10a_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="10a_right_thigh" pos="0 -.1 -.04">
            <joint name="10a_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="10a_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="10a_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="10a_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="10a_right_shin" pos="0 .01 -.403">
              <joint name="10a_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="10a_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="10a_right_foot" pos="0 0 -.39">
                <joint name="10a_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="10a_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="10a_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="10a_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="10a_left_thigh" pos="0 .1 -.04">
            <joint name="10a_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="10a_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="10a_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="10a_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="10a_left_shin" pos="0 -.01 -.403">
              <joint name="10a_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="10a_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="10a_left_foot" pos="0 0 -.39">
                <joint name="10a_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="10a_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="10a_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="10a_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="10a_right_upper_arm" pos="0 -.17 .06">
        <joint name="10a_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="10a_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="10a_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="10a_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="10a_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="10a_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="10a_right_hand" pos=".18 .18 .18">
            <geom name="10a_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="10a_left_upper_arm" pos="0 .17 .06">
        <joint name="10a_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="10a_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="10a_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="10a_left_lower_arm" pos=".18 .18 -.18">
          <joint name="10a_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="10a_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="10a_left_hand" pos=".18 -.18 .18">
            <geom name="10a_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="11a_torso" pos="-1 -5 1.5" childclass="body">
      <camera name="11a_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="11a_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="11a_root"/>
      <geom name="11a_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="11a_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="11a_head" pos="0 0 .19">
        <geom name="11a_head" type="sphere" size=".09"/>
        <camera name="11a_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="11a_lower_waist" pos="-.01 0 -.26">
        <geom name="11a_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="11a_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="11a_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="11a_pelvis" pos="0 0 -.165">
          <joint name="11a_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="11a_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="11a_right_thigh" pos="0 -.1 -.04">
            <joint name="11a_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="11a_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="11a_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="11a_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="11a_right_shin" pos="0 .01 -.403">
              <joint name="11a_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="11a_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="11a_right_foot" pos="0 0 -.39">
                <joint name="11a_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="11a_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="11a_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="11a_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="11a_left_thigh" pos="0 .1 -.04">
            <joint name="11a_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="11a_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="11a_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="11a_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="11a_left_shin" pos="0 -.01 -.403">
              <joint name="11a_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="11a_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="11a_left_foot" pos="0 0 -.39">
                <joint name="11a_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="11a_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="11a_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="11a_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="11a_right_upper_arm" pos="0 -.17 .06">
        <joint name="11a_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="11a_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="11a_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="11a_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="11a_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="11a_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="11a_right_hand" pos=".18 .18 .18">
            <geom name="11a_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="11a_left_upper_arm" pos="0 .17 .06">
        <joint name="11a_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="11a_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="11a_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="11a_left_lower_arm" pos=".18 .18 -.18">
          <joint name="11a_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="11a_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="11a_left_hand" pos=".18 -.18 .18">
            <geom name="11a_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="1b_torso" pos="1 0 1.5" childclass="body" euler="0 0 180">
      <camera name="1b_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="1b_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="1b_root"/>
      <geom name="1b_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="1b_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="1b_head" pos="0 0 .19">
        <geom name="1b_head" type="sphere" size=".09"/>
        <camera name="1b_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="1b_lower_waist" pos="-.01 0 -.26">
        <geom name="1b_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="1b_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="1b_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="1b_pelvis" pos="0 0 -.165">
          <joint name="1b_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="1b_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="1b_right_thigh" pos="0 -.1 -.04">
            <joint name="1b_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="1b_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="1b_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="1b_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="1b_right_shin" pos="0 .01 -.403">
              <joint name="1b_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="1b_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="1b_right_foot" pos="0 0 -.39">
                <joint name="1b_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="1b_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="1b_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="1b_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="1b_left_thigh" pos="0 .1 -.04">
            <joint name="1b_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="1b_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="1b_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="1b_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="1b_left_shin" pos="0 -.01 -.403">
              <joint name="1b_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="1b_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="1b_left_foot" pos="0 0 -.39">
                <joint name="1b_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="1b_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="1b_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="1b_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="1b_right_upper_arm" pos="0 -.17 .06">
        <joint name="1b_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="1b_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="1b_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="1b_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="1b_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="1b_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="1b_right_hand" pos=".18 .18 .18">
            <geom name="1b_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="1b_left_upper_arm" pos="0 .17 .06">
        <joint name="1b_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="1b_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="1b_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="1b_left_lower_arm" pos=".18 .18 -.18">
          <joint name="1b_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="1b_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="1b_left_hand" pos=".18 -.18 .18">
            <geom name="1b_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="2b_torso" pos="1 1 1.5" childclass="body" euler="0 0 180">
      <camera name="2b_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="2b_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="2b_root"/>
      <geom name="2b_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="2b_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="2b_head" pos="0 0 .19">
        <geom name="2b_head" type="sphere" size=".09"/>
        <camera name="2b_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="2b_lower_waist" pos="-.01 0 -.26">
        <geom name="2b_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="2b_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="2b_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="2b_pelvis" pos="0 0 -.165">
          <joint name="2b_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="2b_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="2b_right_thigh" pos="0 -.1 -.04">
            <joint name="2b_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="2b_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="2b_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="2b_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="2b_right_shin" pos="0 .01 -.403">
              <joint name="2b_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="2b_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="2b_right_foot" pos="0 0 -.39">
                <joint name="2b_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="2b_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="2b_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="2b_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="2b_left_thigh" pos="0 .1 -.04">
            <joint name="2b_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="2b_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="2b_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="2b_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="2b_left_shin" pos="0 -.01 -.403">
              <joint name="2b_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="2b_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="2b_left_foot" pos="0 0 -.39">
                <joint name="2b_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="2b_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="2b_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="2b_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="2b_right_upper_arm" pos="0 -.17 .06">
        <joint name="2b_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="2b_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="2b_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="2b_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="2b_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="2b_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="2b_right_hand" pos=".18 .18 .18">
            <geom name="2b_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="2b_left_upper_arm" pos="0 .17 .06">
        <joint name="2b_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="2b_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="2b_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="2b_left_lower_arm" pos=".18 .18 -.18">
          <joint name="2b_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="2b_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="2b_left_hand" pos=".18 -.18 .18">
            <geom name="2b_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="3b_torso" pos="1 -1 1.5" childclass="body" euler="0 0 180">
      <camera name="3b_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="3b_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="3b_root"/>
      <geom name="3b_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="3b_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="3b_head" pos="0 0 .19">
        <geom name="3b_head" type="sphere" size=".09"/>
        <camera name="3b_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="3b_lower_waist" pos="-.01 0 -.26">
        <geom name="3b_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="3b_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="3b_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="3b_pelvis" pos="0 0 -.165">
          <joint name="3b_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="3b_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="3b_right_thigh" pos="0 -.1 -.04">
            <joint name="3b_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="3b_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="3b_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="3b_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="3b_right_shin" pos="0 .01 -.403">
              <joint name="3b_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="3b_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="3b_right_foot" pos="0 0 -.39">
                <joint name="3b_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="3b_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="3b_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="3b_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="3b_left_thigh" pos="0 .1 -.04">
            <joint name="3b_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="3b_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="3b_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="3b_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="3b_left_shin" pos="0 -.01 -.403">
              <joint name="3b_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="3b_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="3b_left_foot" pos="0 0 -.39">
                <joint name="3b_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="3b_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="3b_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="3b_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="3b_right_upper_arm" pos="0 -.17 .06">
        <joint name="3b_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="3b_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="3b_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="3b_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="3b_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="3b_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="3b_right_hand" pos=".18 .18 .18">
            <geom name="3b_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="3b_left_upper_arm" pos="0 .17 .06">
        <joint name="3b_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="3b_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="3b_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="3b_left_lower_arm" pos=".18 .18 -.18">
          <joint name="3b_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="3b_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="3b_left_hand" pos=".18 -.18 .18">
            <geom name="3b_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="4b_torso" pos="1 2 1.5" childclass="body" euler="0 0 180">
      <camera name="4b_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="4b_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="4b_root"/>
      <geom name="4b_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="4b_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="4b_head" pos="0 0 .19">
        <geom name="4b_head" type="sphere" size=".09"/>
        <camera name="4b_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="4b_lower_waist" pos="-.01 0 -.26">
        <geom name="4b_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="4b_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="4b_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="4b_pelvis" pos="0 0 -.165">
          <joint name="4b_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="4b_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="4b_right_thigh" pos="0 -.1 -.04">
            <joint name="4b_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="4b_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="4b_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="4b_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="4b_right_shin" pos="0 .01 -.403">
              <joint name="4b_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="4b_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="4b_right_foot" pos="0 0 -.39">
                <joint name="4b_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="4b_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="4b_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="4b_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="4b_left_thigh" pos="0 .1 -.04">
            <joint name="4b_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="4b_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="4b_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="4b_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="4b_left_shin" pos="0 -.01 -.403">
              <joint name="4b_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="4b_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="4b_left_foot" pos="0 0 -.39">
                <joint name="4b_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="4b_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="4b_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="4b_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="4b_right_upper_arm" pos="0 -.17 .06">
        <joint name="4b_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="4b_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="4b_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="4b_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="4b_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="4b_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="4b_right_hand" pos=".18 .18 .18">
            <geom name="4b_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="4b_left_upper_arm" pos="0 .17 .06">
        <joint name="4b_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="4b_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="4b_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="4b_left_lower_arm" pos=".18 .18 -.18">
          <joint name="4b_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="4b_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="4b_left_hand" pos=".18 -.18 .18">
            <geom name="4b_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="5b_torso" pos="1 -2 1.5" childclass="body" euler="0 0 180">
      <camera name="5b_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="5b_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="5b_root"/>
      <geom name="5b_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="5b_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="5b_head" pos="0 0 .19">
        <geom name="5b_head" type="sphere" size=".09"/>
        <camera name="5b_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="5b_lower_waist" pos="-.01 0 -.26">
        <geom name="5b_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="5b_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="5b_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="5b_pelvis" pos="0 0 -.165">
          <joint name="5b_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="5b_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="5b_right_thigh" pos="0 -.1 -.04">
            <joint name="5b_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="5b_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="5b_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="5b_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="5b_right_shin" pos="0 .01 -.403">
              <joint name="5b_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="5b_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="5b_right_foot" pos="0 0 -.39">
                <joint name="5b_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="5b_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="5b_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="5b_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="5b_left_thigh" pos="0 .1 -.04">
            <joint name="5b_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="5b_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="5b_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="5b_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="5b_left_shin" pos="0 -.01 -.403">
              <joint name="5b_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="5b_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="5b_left_foot" pos="0 0 -.39">
                <joint name="5b_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="5b_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="5b_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="5b_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="5b_right_upper_arm" pos="0 -.17 .06">
        <joint name="5b_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="5b_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="5b_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="5b_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="5b_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="5b_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="5b_right_hand" pos=".18 .18 .18">
            <geom name="5b_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="5b_left_upper_arm" pos="0 .17 .06">
        <joint name="5b_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="5b_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="5b_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="5b_left_lower_arm" pos=".18 .18 -.18">
          <joint name="5b_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="5b_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="5b_left_hand" pos=".18 -.18 .18">
            <geom name="5b_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="6b_torso" pos="1 3 1.5" childclass="body" euler="0 0 180">
      <camera name="6b_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="6b_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="6b_root"/>
      <geom name="6b_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="6b_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="6b_head" pos="0 0 .19">
        <geom name="6b_head" type="sphere" size=".09"/>
        <camera name="6b_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="6b_lower_waist" pos="-.01 0 -.26">
        <geom name="6b_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="6b_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="6b_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="6b_pelvis" pos="0 0 -.165">
          <joint name="6b_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="6b_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="6b_right_thigh" pos="0 -.1 -.04">
            <joint name="6b_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="6b_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="6b_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="6b_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="6b_right_shin" pos="0 .01 -.403">
              <joint name="6b_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="6b_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="6b_right_foot" pos="0 0 -.39">
                <joint name="6b_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="6b_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="6b_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="6b_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="6b_left_thigh" pos="0 .1 -.04">
            <joint name="6b_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="6b_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="6b_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="6b_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="6b_left_shin" pos="0 -.01 -.403">
              <joint name="6b_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="6b_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="6b_left_foot" pos="0 0 -.39">
                <joint name="6b_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="6b_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="6b_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="6b_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="6b_right_upper_arm" pos="0 -.17 .06">
        <joint name="6b_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="6b_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="6b_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="6b_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="6b_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="6b_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="6b_right_hand" pos=".18 .18 .18">
            <geom name="6b_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="6b_left_upper_arm" pos="0 .17 .06">
        <joint name="6b_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="6b_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="6b_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="6b_left_lower_arm" pos=".18 .18 -.18">
          <joint name="6b_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="6b_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="6b_left_hand" pos=".18 -.18 .18">
            <geom name="6b_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="7b_torso" pos="1 -3 1.5" childclass="body" euler="0 0 180">
      <camera name="7b_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="7b_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="7b_root"/>
      <geom name="7b_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="7b_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="7b_head" pos="0 0 .19">
        <geom name="7b_head" type="sphere" size=".09"/>
        <camera name="7b_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="7b_lower_waist" pos="-.01 0 -.26">
        <geom name="7b_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="7b_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="7b_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="7b_pelvis" pos="0 0 -.165">
          <joint name="7b_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="7b_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="7b_right_thigh" pos="0 -.1 -.04">
            <joint name="7b_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="7b_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="7b_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="7b_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="7b_right_shin" pos="0 .01 -.403">
              <joint name="7b_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="7b_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="7b_right_foot" pos="0 0 -.39">
                <joint name="7b_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="7b_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="7b_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="7b_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="7b_left_thigh" pos="0 .1 -.04">
            <joint name="7b_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="7b_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="7b_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="7b_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="7b_left_shin" pos="0 -.01 -.403">
              <joint name="7b_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="7b_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="7b_left_foot" pos="0 0 -.39">
                <joint name="7b_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="7b_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="7b_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="7b_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="7b_right_upper_arm" pos="0 -.17 .06">
        <joint name="7b_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="7b_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="7b_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="7b_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="7b_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="7b_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="7b_right_hand" pos=".18 .18 .18">
            <geom name="7b_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="7b_left_upper_arm" pos="0 .17 .06">
        <joint name="7b_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="7b_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="7b_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="7b_left_lower_arm" pos=".18 .18 -.18">
          <joint name="7b_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="7b_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="7b_left_hand" pos=".18 -.18 .18">
            <geom name="7b_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="8b_torso" pos="1 4 1.5" childclass="body" euler="0 0 180">
      <camera name="8b_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="8b_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="8b_root"/>
      <geom name="8b_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="8b_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="8b_head" pos="0 0 .19">
        <geom name="8b_head" type="sphere" size=".09"/>
        <camera name="8b_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="8b_lower_waist" pos="-.01 0 -.26">
        <geom name="8b_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="8b_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="8b_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="8b_pelvis" pos="0 0 -.165">
          <joint name="8b_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="8b_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="8b_right_thigh" pos="0 -.1 -.04">
            <joint name="8b_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="8b_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="8b_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="8b_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="8b_right_shin" pos="0 .01 -.403">
              <joint name="8b_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="8b_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="8b_right_foot" pos="0 0 -.39">
                <joint name="8b_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="8b_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="8b_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="8b_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="8b_left_thigh" pos="0 .1 -.04">
            <joint name="8b_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="8b_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="8b_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="8b_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="8b_left_shin" pos="0 -.01 -.403">
              <joint name="8b_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="8b_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="8b_left_foot" pos="0 0 -.39">
                <joint name="8b_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="8b_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="8b_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="8b_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="8b_right_upper_arm" pos="0 -.17 .06">
        <joint name="8b_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="8b_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="8b_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="8b_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="8b_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="8b_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="8b_right_hand" pos=".18 .18 .18">
            <geom name="8b_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="8b_left_upper_arm" pos="0 .17 .06">
        <joint name="8b_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="8b_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="8b_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="8b_left_lower_arm" pos=".18 .18 -.18">
          <joint name="8b_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="8b_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="8b_left_hand" pos=".18 -.18 .18">
            <geom name="8b_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="9b_torso" pos="1 -4 1.5" childclass="body" euler="0 0 180">
      <camera name="9b_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="9b_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="9b_root"/>
      <geom name="9b_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="9b_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="9b_head" pos="0 0 .19">
        <geom name="9b_head" type="sphere" size=".09"/>
        <camera name="9b_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="9b_lower_waist" pos="-.01 0 -.26">
        <geom name="9b_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="9b_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="9b_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="9b_pelvis" pos="0 0 -.165">
          <joint name="9b_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="9b_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="9b_right_thigh" pos="0 -.1 -.04">
            <joint name="9b_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="9b_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="9b_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="9b_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="9b_right_shin" pos="0 .01 -.403">
              <joint name="9b_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="9b_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="9b_right_foot" pos="0 0 -.39">
                <joint name="9b_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="9b_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="9b_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="9b_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="9b_left_thigh" pos="0 .1 -.04">
            <joint name="9b_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="9b_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="9b_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="9b_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="9b_left_shin" pos="0 -.01 -.403">
              <joint name="9b_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="9b_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="9b_left_foot" pos="0 0 -.39">
                <joint name="9b_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="9b_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="9b_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="9b_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="9b_right_upper_arm" pos="0 -.17 .06">
        <joint name="9b_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="9b_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="9b_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="9b_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="9b_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="9b_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="9b_right_hand" pos=".18 .18 .18">
            <geom name="9b_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="9b_left_upper_arm" pos="0 .17 .06">
        <joint name="9b_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="9b_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="9b_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="9b_left_lower_arm" pos=".18 .18 -.18">
          <joint name="9b_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="9b_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="9b_left_hand" pos=".18 -.18 .18">
            <geom name="9b_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="10b_torso" pos="1 5 1.5" childclass="body" euler="0 0 180">
      <camera name="10b_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="10b_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="10b_root"/>
      <geom name="10b_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="10b_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="10b_head" pos="0 0 .19">
        <geom name="10b_head" type="sphere" size=".09"/>
        <camera name="10b_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="10b_lower_waist" pos="-.01 0 -.26">
        <geom name="10b_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="10b_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="10b_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="10b_pelvis" pos="0 0 -.165">
          <joint name="10b_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="10b_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="10b_right_thigh" pos="0 -.1 -.04">
            <joint name="10b_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="10b_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="10b_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="10b_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="10b_right_shin" pos="0 .01 -.403">
              <joint name="10b_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="10b_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="10b_right_foot" pos="0 0 -.39">
                <joint name="10b_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="10b_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="10b_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="10b_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="10b_left_thigh" pos="0 .1 -.04">
            <joint name="10b_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="10b_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="10b_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="10b_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="10b_left_shin" pos="0 -.01 -.403">
              <joint name="10b_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="10b_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="10b_left_foot" pos="0 0 -.39">
                <joint name="10b_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="10b_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="10b_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="10b_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="10b_right_upper_arm" pos="0 -.17 .06">
        <joint name="10b_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="10b_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="10b_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="10b_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="10b_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="10b_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="10b_right_hand" pos=".18 .18 .18">
            <geom name="10b_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="10b_left_upper_arm" pos="0 .17 .06">
        <joint name="10b_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="10b_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="10b_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="10b_left_lower_arm" pos=".18 .18 -.18">
          <joint name="10b_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="10b_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="10b_left_hand" pos=".18 -.18 .18">
            <geom name="10b_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

    <body name="11b_torso" pos="1 -5 1.5" childclass="body" euler="0 0 180">
      <camera name="11b_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="11b_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="11b_root"/>
      <geom name="11b_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="11b_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="11b_head" pos="0 0 .19">
        <geom name="11b_head" type="sphere" size=".09"/>
        <camera name="11b_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="11b_lower_waist" pos="-.01 0 -.26">
        <geom name="11b_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="11b_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="11b_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="11b_pelvis" pos="0 0 -.165">
          <joint name="11b_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="11b_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="11b_right_thigh" pos="0 -.1 -.04">
            <joint name="11b_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="11b_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="11b_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="11b_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="11b_right_shin" pos="0 .01 -.403">
              <joint name="11b_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="11b_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="11b_right_foot" pos="0 0 -.39">
                <joint name="11b_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="11b_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="11b_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="11b_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="11b_left_thigh" pos="0 .1 -.04">
            <joint name="11b_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="11b_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="11b_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="11b_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="11b_left_shin" pos="0 -.01 -.403">
              <joint name="11b_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="11b_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="11b_left_foot" pos="0 0 -.39">
                <joint name="11b_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="11b_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="11b_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="11b_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="11b_right_upper_arm" pos="0 -.17 .06">
        <joint name="11b_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="11b_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="11b_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="11b_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="11b_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="11b_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="11b_right_hand" pos=".18 .18 .18">
            <geom name="11b_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="11b_left_upper_arm" pos="0 .17 .06">
        <joint name="11b_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="11b_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="11b_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="11b_left_lower_arm" pos=".18 .18 -.18">
          <joint name="11b_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="11b_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="11b_left_hand" pos=".18 -.18 .18">
            <geom name="11b_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor gear="40"  joint="1a_abdomen_y"/>
    <motor gear="40"  joint="1a_abdomen_z"/>
    <motor gear="40"  joint="1a_abdomen_x"/>
    <motor gear="40"  joint="1a_right_hip_x"/>
    <motor gear="40"  joint="1a_right_hip_z"/>
    <motor gear="120" joint="1a_right_hip_y"/>
    <motor gear="80"  joint="1a_right_knee"/>
    <motor gear="20"  joint="1a_right_ankle_x"/>
    <motor gear="20"  joint="1a_right_ankle_y"/>
    <motor gear="40"  joint="1a_left_hip_x"/>
    <motor gear="40"  joint="1a_left_hip_z"/>
    <motor gear="120" joint="1a_left_hip_y"/>
    <motor gear="80"  joint="1a_left_knee"/>
    <motor gear="20"  joint="1a_left_ankle_x"/>
    <motor gear="20"  joint="1a_left_ankle_y"/>
    <motor gear="20"  joint="1a_right_shoulder1"/>
    <motor gear="20"  joint="1a_right_shoulder2"/>
    <motor gear="40"  joint="1a_right_elbow"/>
    <motor gear="20"  joint="1a_left_shoulder1"/>
    <motor gear="20"  joint="1a_left_shoulder2"/>
    <motor gear="40"  joint="1a_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="2a_abdomen_y"/>
    <motor gear="40"  joint="2a_abdomen_z"/>
    <motor gear="40"  joint="2a_abdomen_x"/>
    <motor gear="40"  joint="2a_right_hip_x"/>
    <motor gear="40"  joint="2a_right_hip_z"/>
    <motor gear="120" joint="2a_right_hip_y"/>
    <motor gear="80"  joint="2a_right_knee"/>
    <motor gear="20"  joint="2a_right_ankle_x"/>
    <motor gear="20"  joint="2a_right_ankle_y"/>
    <motor gear="40"  joint="2a_left_hip_x"/>
    <motor gear="40"  joint="2a_left_hip_z"/>
    <motor gear="120" joint="2a_left_hip_y"/>
    <motor gear="80"  joint="2a_left_knee"/>
    <motor gear="20"  joint="2a_left_ankle_x"/>
    <motor gear="20"  joint="2a_left_ankle_y"/>
    <motor gear="20"  joint="2a_right_shoulder1"/>
    <motor gear="20"  joint="2a_right_shoulder2"/>
    <motor gear="40"  joint="2a_right_elbow"/>
    <motor gear="20"  joint="2a_left_shoulder1"/>
    <motor gear="20"  joint="2a_left_shoulder2"/>
    <motor gear="40"  joint="2a_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="3a_abdomen_y"/>
    <motor gear="40"  joint="3a_abdomen_z"/>
    <motor gear="40"  joint="3a_abdomen_x"/>
    <motor gear="40"  joint="3a_right_hip_x"/>
    <motor gear="40"  joint="3a_right_hip_z"/>
    <motor gear="120" joint="3a_right_hip_y"/>
    <motor gear="80"  joint="3a_right_knee"/>
    <motor gear="20"  joint="3a_right_ankle_x"/>
    <motor gear="20"  joint="3a_right_ankle_y"/>
    <motor gear="40"  joint="3a_left_hip_x"/>
    <motor gear="40"  joint="3a_left_hip_z"/>
    <motor gear="120" joint="3a_left_hip_y"/>
    <motor gear="80"  joint="3a_left_knee"/>
    <motor gear="20"  joint="3a_left_ankle_x"/>
    <motor gear="20"  joint="3a_left_ankle_y"/>
    <motor gear="20"  joint="3a_right_shoulder1"/>
    <motor gear="20"  joint="3a_right_shoulder2"/>
    <motor gear="40"  joint="3a_right_elbow"/>
    <motor gear="20"  joint="3a_left_shoulder1"/>
    <motor gear="20"  joint="3a_left_shoulder2"/>
    <motor gear="40"  joint="3a_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="4a_abdomen_y"/>
    <motor gear="40"  joint="4a_abdomen_z"/>
    <motor gear="40"  joint="4a_abdomen_x"/>
    <motor gear="40"  joint="4a_right_hip_x"/>
    <motor gear="40"  joint="4a_right_hip_z"/>
    <motor gear="120" joint="4a_right_hip_y"/>
    <motor gear="80"  joint="4a_right_knee"/>
    <motor gear="20"  joint="4a_right_ankle_x"/>
    <motor gear="20"  joint="4a_right_ankle_y"/>
    <motor gear="40"  joint="4a_left_hip_x"/>
    <motor gear="40"  joint="4a_left_hip_z"/>
    <motor gear="120" joint="4a_left_hip_y"/>
    <motor gear="80"  joint="4a_left_knee"/>
    <motor gear="20"  joint="4a_left_ankle_x"/>
    <motor gear="20"  joint="4a_left_ankle_y"/>
    <motor gear="20"  joint="4a_right_shoulder1"/>
    <motor gear="20"  joint="4a_right_shoulder2"/>
    <motor gear="40"  joint="4a_right_elbow"/>
    <motor gear="20"  joint="4a_left_shoulder1"/>
    <motor gear="20"  joint="4a_left_shoulder2"/>
    <motor gear="40"  joint="4a_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="5a_abdomen_y"/>
    <motor gear="40"  joint="5a_abdomen_z"/>
    <motor gear="40"  joint="5a_abdomen_x"/>
    <motor gear="40"  joint="5a_right_hip_x"/>
    <motor gear="40"  joint="5a_right_hip_z"/>
    <motor gear="120" joint="5a_right_hip_y"/>
    <motor gear="80"  joint="5a_right_knee"/>
    <motor gear="20"  joint="5a_right_ankle_x"/>
    <motor gear="20"  joint="5a_right_ankle_y"/>
    <motor gear="40"  joint="5a_left_hip_x"/>
    <motor gear="40"  joint="5a_left_hip_z"/>
    <motor gear="120" joint="5a_left_hip_y"/>
    <motor gear="80"  joint="5a_left_knee"/>
    <motor gear="20"  joint="5a_left_ankle_x"/>
    <motor gear="20"  joint="5a_left_ankle_y"/>
    <motor gear="20"  joint="5a_right_shoulder1"/>
    <motor gear="20"  joint="5a_right_shoulder2"/>
    <motor gear="40"  joint="5a_right_elbow"/>
    <motor gear="20"  joint="5a_left_shoulder1"/>
    <motor gear="20"  joint="5a_left_shoulder2"/>
    <motor gear="40"  joint="5a_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="6a_abdomen_y"/>
    <motor gear="40"  joint="6a_abdomen_z"/>
    <motor gear="40"  joint="6a_abdomen_x"/>
    <motor gear="40"  joint="6a_right_hip_x"/>
    <motor gear="40"  joint="6a_right_hip_z"/>
    <motor gear="120" joint="6a_right_hip_y"/>
    <motor gear="80"  joint="6a_right_knee"/>
    <motor gear="20"  joint="6a_right_ankle_x"/>
    <motor gear="20"  joint="6a_right_ankle_y"/>
    <motor gear="40"  joint="6a_left_hip_x"/>
    <motor gear="40"  joint="6a_left_hip_z"/>
    <motor gear="120" joint="6a_left_hip_y"/>
    <motor gear="80"  joint="6a_left_knee"/>
    <motor gear="20"  joint="6a_left_ankle_x"/>
    <motor gear="20"  joint="6a_left_ankle_y"/>
    <motor gear="20"  joint="6a_right_shoulder1"/>
    <motor gear="20"  joint="6a_right_shoulder2"/>
    <motor gear="40"  joint="6a_right_elbow"/>
    <motor gear="20"  joint="6a_left_shoulder1"/>
    <motor gear="20"  joint="6a_left_shoulder2"/>
    <motor gear="40"  joint="6a_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="7a_abdomen_y"/>
    <motor gear="40"  joint="7a_abdomen_z"/>
    <motor gear="40"  joint="7a_abdomen_x"/>
    <motor gear="40"  joint="7a_right_hip_x"/>
    <motor gear="40"  joint="7a_right_hip_z"/>
    <motor gear="120" joint="7a_right_hip_y"/>
    <motor gear="80"  joint="7a_right_knee"/>
    <motor gear="20"  joint="7a_right_ankle_x"/>
    <motor gear="20"  joint="7a_right_ankle_y"/>
    <motor gear="40"  joint="7a_left_hip_x"/>
    <motor gear="40"  joint="7a_left_hip_z"/>
    <motor gear="120" joint="7a_left_hip_y"/>
    <motor gear="80"  joint="7a_left_knee"/>
    <motor gear="20"  joint="7a_left_ankle_x"/>
    <motor gear="20"  joint="7a_left_ankle_y"/>
    <motor gear="20"  joint="7a_right_shoulder1"/>
    <motor gear="20"  joint="7a_right_shoulder2"/>
    <motor gear="40"  joint="7a_right_elbow"/>
    <motor gear="20"  joint="7a_left_shoulder1"/>
    <motor gear="20"  joint="7a_left_shoulder2"/>
    <motor gear="40"  joint="7a_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="8a_abdomen_y"/>
    <motor gear="40"  joint="8a_abdomen_z"/>
    <motor gear="40"  joint="8a_abdomen_x"/>
    <motor gear="40"  joint="8a_right_hip_x"/>
    <motor gear="40"  joint="8a_right_hip_z"/>
    <motor gear="120" joint="8a_right_hip_y"/>
    <motor gear="80"  joint="8a_right_knee"/>
    <motor gear="20"  joint="8a_right_ankle_x"/>
    <motor gear="20"  joint="8a_right_ankle_y"/>
    <motor gear="40"  joint="8a_left_hip_x"/>
    <motor gear="40"  joint="8a_left_hip_z"/>
    <motor gear="120" joint="8a_left_hip_y"/>
    <motor gear="80"  joint="8a_left_knee"/>
    <motor gear="20"  joint="8a_left_ankle_x"/>
    <motor gear="20"  joint="8a_left_ankle_y"/>
    <motor gear="20"  joint="8a_right_shoulder1"/>
    <motor gear="20"  joint="8a_right_shoulder2"/>
    <motor gear="40"  joint="8a_right_elbow"/>
    <motor gear="20"  joint="8a_left_shoulder1"/>
    <motor gear="20"  joint="8a_left_shoulder2"/>
    <motor gear="40"  joint="8a_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="9a_abdomen_y"/>
    <motor gear="40"  joint="9a_abdomen_z"/>
    <motor gear="40"  joint="9a_abdomen_x"/>
    <motor gear="40"  joint="9a_right_hip_x"/>
    <motor gear="40"  joint="9a_right_hip_z"/>
    <motor gear="120" joint="9a_right_hip_y"/>
    <motor gear="80"  joint="9a_right_knee"/>
    <motor gear="20"  joint="9a_right_ankle_x"/>
    <motor gear="20"  joint="9a_right_ankle_y"/>
    <motor gear="40"  joint="9a_left_hip_x"/>
    <motor gear="40"  joint="9a_left_hip_z"/>
    <motor gear="120" joint="9a_left_hip_y"/>
    <motor gear="80"  joint="9a_left_knee"/>
    <motor gear="20"  joint="9a_left_ankle_x"/>
    <motor gear="20"  joint="9a_left_ankle_y"/>
    <motor gear="20"  joint="9a_right_shoulder1"/>
    <motor gear="20"  joint="9a_right_shoulder2"/>
    <motor gear="40"  joint="9a_right_elbow"/>
    <motor gear="20"  joint="9a_left_shoulder1"/>
    <motor gear="20"  joint="9a_left_shoulder2"/>
    <motor gear="40"  joint="9a_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="10a_abdomen_y"/>
    <motor gear="40"  joint="10a_abdomen_z"/>
    <motor gear="40"  joint="10a_abdomen_x"/>
    <motor gear="40"  joint="10a_right_hip_x"/>
    <motor gear="40"  joint="10a_right_hip_z"/>
    <motor gear="120" joint="10a_right_hip_y"/>
    <motor gear="80"  joint="10a_right_knee"/>
    <motor gear="20"  joint="10a_right_ankle_x"/>
    <motor gear="20"  joint="10a_right_ankle_y"/>
    <motor gear="40"  joint="10a_left_hip_x"/>
    <motor gear="40"  joint="10a_left_hip_z"/>
    <motor gear="120" joint="10a_left_hip_y"/>
    <motor gear="80"  joint="10a_left_knee"/>
    <motor gear="20"  joint="10a_left_ankle_x"/>
    <motor gear="20"  joint="10a_left_ankle_y"/>
    <motor gear="20"  joint="10a_right_shoulder1"/>
    <motor gear="20"  joint="10a_right_shoulder2"/>
    <motor gear="40"  joint="10a_right_elbow"/>
    <motor gear="20"  joint="10a_left_shoulder1"/>
    <motor gear="20"  joint="10a_left_shoulder2"/>
    <motor gear="40"  joint="10a_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="11a_abdomen_y"/>
    <motor gear="40"  joint="11a_abdomen_z"/>
    <motor gear="40"  joint="11a_abdomen_x"/>
    <motor gear="40"  joint="11a_right_hip_x"/>
    <motor gear="40"  joint="11a_right_hip_z"/>
    <motor gear="120" joint="11a_right_hip_y"/>
    <motor gear="80"  joint="11a_right_knee"/>
    <motor gear="20"  joint="11a_right_ankle_x"/>
    <motor gear="20"  joint="11a_right_ankle_y"/>
    <motor gear="40"  joint="11a_left_hip_x"/>
    <motor gear="40"  joint="11a_left_hip_z"/>
    <motor gear="120" joint="11a_left_hip_y"/>
    <motor gear="80"  joint="11a_left_knee"/>
    <motor gear="20"  joint="11a_left_ankle_x"/>
    <motor gear="20"  joint="11a_left_ankle_y"/>
    <motor gear="20"  joint="11a_right_shoulder1"/>
    <motor gear="20"  joint="11a_right_shoulder2"/>
    <motor gear="40"  joint="11a_right_elbow"/>
    <motor gear="20"  joint="11a_left_shoulder1"/>
    <motor gear="20"  joint="11a_left_shoulder2"/>
    <motor gear="40"  joint="11a_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="1b_abdomen_y"/>
    <motor gear="40"  joint="1b_abdomen_z"/>
    <motor gear="40"  joint="1b_abdomen_x"/>
    <motor gear="40"  joint="1b_right_hip_x"/>
    <motor gear="40"  joint="1b_right_hip_z"/>
    <motor gear="120" joint="1b_right_hip_y"/>
    <motor gear="80"  joint="1b_right_knee"/>
    <motor gear="20"  joint="1b_right_ankle_x"/>
    <motor gear="20"  joint="1b_right_ankle_y"/>
    <motor gear="40"  joint="1b_left_hip_x"/>
    <motor gear="40"  joint="1b_left_hip_z"/>
    <motor gear="120" joint="1b_left_hip_y"/>
    <motor gear="80"  joint="1b_left_knee"/>
    <motor gear="20"  joint="1b_left_ankle_x"/>
    <motor gear="20"  joint="1b_left_ankle_y"/>
    <motor gear="20"  joint="1b_right_shoulder1"/>
    <motor gear="20"  joint="1b_right_shoulder2"/>
    <motor gear="40"  joint="1b_right_elbow"/>
    <motor gear="20"  joint="1b_left_shoulder1"/>
    <motor gear="20"  joint="1b_left_shoulder2"/>
    <motor gear="40"  joint="1b_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="2b_abdomen_y"/>
    <motor gear="40"  joint="2b_abdomen_z"/>
    <motor gear="40"  joint="2b_abdomen_x"/>
    <motor gear="40"  joint="2b_right_hip_x"/>
    <motor gear="40"  joint="2b_right_hip_z"/>
    <motor gear="120" joint="2b_right_hip_y"/>
    <motor gear="80"  joint="2b_right_knee"/>
    <motor gear="20"  joint="2b_right_ankle_x"/>
    <motor gear="20"  joint="2b_right_ankle_y"/>
    <motor gear="40"  joint="2b_left_hip_x"/>
    <motor gear="40"  joint="2b_left_hip_z"/>
    <motor gear="120" joint="2b_left_hip_y"/>
    <motor gear="80"  joint="2b_left_knee"/>
    <motor gear="20"  joint="2b_left_ankle_x"/>
    <motor gear="20"  joint="2b_left_ankle_y"/>
    <motor gear="20"  joint="2b_right_shoulder1"/>
    <motor gear="20"  joint="2b_right_shoulder2"/>
    <motor gear="40"  joint="2b_right_elbow"/>
    <motor gear="20"  joint="2b_left_shoulder1"/>
    <motor gear="20"  joint="2b_left_shoulder2"/>
    <motor gear="40"  joint="2b_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="3b_abdomen_y"/>
    <motor gear="40"  joint="3b_abdomen_z"/>
    <motor gear="40"  joint="3b_abdomen_x"/>
    <motor gear="40"  joint="3b_right_hip_x"/>
    <motor gear="40"  joint="3b_right_hip_z"/>
    <motor gear="120" joint="3b_right_hip_y"/>
    <motor gear="80"  joint="3b_right_knee"/>
    <motor gear="20"  joint="3b_right_ankle_x"/>
    <motor gear="20"  joint="3b_right_ankle_y"/>
    <motor gear="40"  joint="3b_left_hip_x"/>
    <motor gear="40"  joint="3b_left_hip_z"/>
    <motor gear="120" joint="3b_left_hip_y"/>
    <motor gear="80"  joint="3b_left_knee"/>
    <motor gear="20"  joint="3b_left_ankle_x"/>
    <motor gear="20"  joint="3b_left_ankle_y"/>
    <motor gear="20"  joint="3b_right_shoulder1"/>
    <motor gear="20"  joint="3b_right_shoulder2"/>
    <motor gear="40"  joint="3b_right_elbow"/>
    <motor gear="20"  joint="3b_left_shoulder1"/>
    <motor gear="20"  joint="3b_left_shoulder2"/>
    <motor gear="40"  joint="3b_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="4b_abdomen_y"/>
    <motor gear="40"  joint="4b_abdomen_z"/>
    <motor gear="40"  joint="4b_abdomen_x"/>
    <motor gear="40"  joint="4b_right_hip_x"/>
    <motor gear="40"  joint="4b_right_hip_z"/>
    <motor gear="120" joint="4b_right_hip_y"/>
    <motor gear="80"  joint="4b_right_knee"/>
    <motor gear="20"  joint="4b_right_ankle_x"/>
    <motor gear="20"  joint="4b_right_ankle_y"/>
    <motor gear="40"  joint="4b_left_hip_x"/>
    <motor gear="40"  joint="4b_left_hip_z"/>
    <motor gear="120" joint="4b_left_hip_y"/>
    <motor gear="80"  joint="4b_left_knee"/>
    <motor gear="20"  joint="4b_left_ankle_x"/>
    <motor gear="20"  joint="4b_left_ankle_y"/>
    <motor gear="20"  joint="4b_right_shoulder1"/>
    <motor gear="20"  joint="4b_right_shoulder2"/>
    <motor gear="40"  joint="4b_right_elbow"/>
    <motor gear="20"  joint="4b_left_shoulder1"/>
    <motor gear="20"  joint="4b_left_shoulder2"/>
    <motor gear="40"  joint="4b_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="5b_abdomen_y"/>
    <motor gear="40"  joint="5b_abdomen_z"/>
    <motor gear="40"  joint="5b_abdomen_x"/>
    <motor gear="40"  joint="5b_right_hip_x"/>
    <motor gear="40"  joint="5b_right_hip_z"/>
    <motor gear="120" joint="5b_right_hip_y"/>
    <motor gear="80"  joint="5b_right_knee"/>
    <motor gear="20"  joint="5b_right_ankle_x"/>
    <motor gear="20"  joint="5b_right_ankle_y"/>
    <motor gear="40"  joint="5b_left_hip_x"/>
    <motor gear="40"  joint="5b_left_hip_z"/>
    <motor gear="120" joint="5b_left_hip_y"/>
    <motor gear="80"  joint="5b_left_knee"/>
    <motor gear="20"  joint="5b_left_ankle_x"/>
    <motor gear="20"  joint="5b_left_ankle_y"/>
    <motor gear="20"  joint="5b_right_shoulder1"/>
    <motor gear="20"  joint="5b_right_shoulder2"/>
    <motor gear="40"  joint="5b_right_elbow"/>
    <motor gear="20"  joint="5b_left_shoulder1"/>
    <motor gear="20"  joint="5b_left_shoulder2"/>
    <motor gear="40"  joint="5b_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="6b_abdomen_y"/>
    <motor gear="40"  joint="6b_abdomen_z"/>
    <motor gear="40"  joint="6b_abdomen_x"/>
    <motor gear="40"  joint="6b_right_hip_x"/>
    <motor gear="40"  joint="6b_right_hip_z"/>
    <motor gear="120" joint="6b_right_hip_y"/>
    <motor gear="80"  joint="6b_right_knee"/>
    <motor gear="20"  joint="6b_right_ankle_x"/>
    <motor gear="20"  joint="6b_right_ankle_y"/>
    <motor gear="40"  joint="6b_left_hip_x"/>
    <motor gear="40"  joint="6b_left_hip_z"/>
    <motor gear="120" joint="6b_left_hip_y"/>
    <motor gear="80"  joint="6b_left_knee"/>
    <motor gear="20"  joint="6b_left_ankle_x"/>
    <motor gear="20"  joint="6b_left_ankle_y"/>
    <motor gear="20"  joint="6b_right_shoulder1"/>
    <motor gear="20"  joint="6b_right_shoulder2"/>
    <motor gear="40"  joint="6b_right_elbow"/>
    <motor gear="20"  joint="6b_left_shoulder1"/>
    <motor gear="20"  joint="6b_left_shoulder2"/>
    <motor gear="40"  joint="6b_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="7b_abdomen_y"/>
    <motor gear="40"  joint="7b_abdomen_z"/>
    <motor gear="40"  joint="7b_abdomen_x"/>
    <motor gear="40"  joint="7b_right_hip_x"/>
    <motor gear="40"  joint="7b_right_hip_z"/>
    <motor gear="120" joint="7b_right_hip_y"/>
    <motor gear="80"  joint="7b_right_knee"/>
    <motor gear="20"  joint="7b_right_ankle_x"/>
    <motor gear="20"  joint="7b_right_ankle_y"/>
    <motor gear="40"  joint="7b_left_hip_x"/>
    <motor gear="40"  joint="7b_left_hip_z"/>
    <motor gear="120" joint="7b_left_hip_y"/>
    <motor gear="80"  joint="7b_left_knee"/>
    <motor gear="20"  joint="7b_left_ankle_x"/>
    <motor gear="20"  joint="7b_left_ankle_y"/>
    <motor gear="20"  joint="7b_right_shoulder1"/>
    <motor gear="20"  joint="7b_right_shoulder2"/>
    <motor gear="40"  joint="7b_right_elbow"/>
    <motor gear="20"  joint="7b_left_shoulder1"/>
    <motor gear="20"  joint="7b_left_shoulder2"/>
    <motor gear="40"  joint="7b_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="8b_abdomen_y"/>
    <motor gear="40"  joint="8b_abdomen_z"/>
    <motor gear="40"  joint="8b_abdomen_x"/>
    <motor gear="40"  joint="8b_right_hip_x"/>
    <motor gear="40"  joint="8b_right_hip_z"/>
    <motor gear="120" joint="8b_right_hip_y"/>
    <motor gear="80"  joint="8b_right_knee"/>
    <motor gear="20"  joint="8b_right_ankle_x"/>
    <motor gear="20"  joint="8b_right_ankle_y"/>
    <motor gear="40"  joint="8b_left_hip_x"/>
    <motor gear="40"  joint="8b_left_hip_z"/>
    <motor gear="120" joint="8b_left_hip_y"/>
    <motor gear="80"  joint="8b_left_knee"/>
    <motor gear="20"  joint="8b_left_ankle_x"/>
    <motor gear="20"  joint="8b_left_ankle_y"/>
    <motor gear="20"  joint="8b_right_shoulder1"/>
    <motor gear="20"  joint="8b_right_shoulder2"/>
    <motor gear="40"  joint="8b_right_elbow"/>
    <motor gear="20"  joint="8b_left_shoulder1"/>
    <motor gear="20"  joint="8b_left_shoulder2"/>
    <motor gear="40"  joint="8b_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="9b_abdomen_y"/>
    <motor gear="40"  joint="9b_abdomen_z"/>
    <motor gear="40"  joint="9b_abdomen_x"/>
    <motor gear="40"  joint="9b_right_hip_x"/>
    <motor gear="40"  joint="9b_right_hip_z"/>
    <motor gear="120" joint="9b_right_hip_y"/>
    <motor gear="80"  joint="9b_right_knee"/>
    <motor gear="20"  joint="9b_right_ankle_x"/>
    <motor gear="20"  joint="9b_right_ankle_y"/>
    <motor gear="40"  joint="9b_left_hip_x"/>
    <motor gear="40"  joint="9b_left_hip_z"/>
    <motor gear="120" joint="9b_left_hip_y"/>
    <motor gear="80"  joint="9b_left_knee"/>
    <motor gear="20"  joint="9b_left_ankle_x"/>
    <motor gear="20"  joint="9b_left_ankle_y"/>
    <motor gear="20"  joint="9b_right_shoulder1"/>
    <motor gear="20"  joint="9b_right_shoulder2"/>
    <motor gear="40"  joint="9b_right_elbow"/>
    <motor gear="20"  joint="9b_left_shoulder1"/>
    <motor gear="20"  joint="9b_left_shoulder2"/>
    <motor gear="40"  joint="9b_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="10b_abdomen_y"/>
    <motor gear="40"  joint="10b_abdomen_z"/>
    <motor gear="40"  joint="10b_abdomen_x"/>
    <motor gear="40"  joint="10b_right_hip_x"/>
    <motor gear="40"  joint="10b_right_hip_z"/>
    <motor gear="120" joint="10b_right_hip_y"/>
    <motor gear="80"  joint="10b_right_knee"/>
    <motor gear="20"  joint="10b_right_ankle_x"/>
    <motor gear="20"  joint="10b_right_ankle_y"/>
    <motor gear="40"  joint="10b_left_hip_x"/>
    <motor gear="40"  joint="10b_left_hip_z"/>
    <motor gear="120" joint="10b_left_hip_y"/>
    <motor gear="80"  joint="10b_left_knee"/>
    <motor gear="20"  joint="10b_left_ankle_x"/>
    <motor gear="20"  joint="10b_left_ankle_y"/>
    <motor gear="20"  joint="10b_right_shoulder1"/>
    <motor gear="20"  joint="10b_right_shoulder2"/>
    <motor gear="40"  joint="10b_right_elbow"/>
    <motor gear="20"  joint="10b_left_shoulder1"/>
    <motor gear="20"  joint="10b_left_shoulder2"/>
    <motor gear="40"  joint="10b_left_elbow"/>
  </actuator>

  <actuator>
    <motor gear="40"  joint="11b_abdomen_y"/>
    <motor gear="40"  joint="11b_abdomen_z"/>
    <motor gear="40"  joint="11b_abdomen_x"/>
    <motor gear="40"  joint="11b_right_hip_x"/>
    <motor gear="40"  joint="11b_right_hip_z"/>
    <motor gear="120" joint="11b_right_hip_y"/>
    <motor gear="80"  joint="11b_right_knee"/>
    <motor gear="20"  joint="11b_right_ankle_x"/>
    <motor gear="20"  joint="11b_right_ankle_y"/>
    <motor gear="40"  joint="11b_left_hip_x"/>
    <motor gear="40"  joint="11b_left_hip_z"/>
    <motor gear="120" joint="11b_left_hip_y"/>
    <motor gear="80"  joint="11b_left_knee"/>
    <motor gear="20"  joint="11b_left_ankle_x"/>
    <motor gear="20"  joint="11b_left_ankle_y"/>
    <motor gear="20"  joint="11b_right_shoulder1"/>
    <motor gear="20"  joint="11b_right_shoulder2"/>
    <motor gear="40"  joint="11b_right_elbow"/>
    <motor gear="20"  joint="11b_left_shoulder1"/>
    <motor gear="20"  joint="11b_left_shoulder2"/>
    <motor gear="40"  joint="11b_left_elbow"/>
  </actuator>
</mujoco>

"""
physics = mujoco.Physics.from_xml_string(swinging_body)
# Visualize the joint axis.
scene_option = mujoco.wrapper.core.MjvOption()
scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True
pixels = physics.render(scene_option=scene_option)

okay = PIL.Image.fromarray(pixels)
# okay.show()
# @title Making a video {vertical-output: true}

duration = 3  # (seconds)
framerate = 60  # (Hz)

# Visualize the joint axis
scene_option = mujoco.wrapper.core.MjvOption()
scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True


print(manimation.writers.list())

# # Simulate and display video.
frames = []
video_name = "humanoids_22"
physics.reset()  # Reset state and time
while physics.data.time < duration:
    physics.step()
    if len(frames) < physics.data.time * framerate:
        pixels = physics.render(scene_option=scene_option)
        frames.append(pixels)
okay = display_video(frames, framerate, video_name)

scene_option = mujoco.wrapper.core.MjvOption()
scene_option.frame = enums.mjtFrame.mjFRAME_GEOM
scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True
pixels = physics.render(scene_option=scene_option)
okay = PIL.Image.fromarray(pixels)
# okay.show()

# @title The "tippe-top" model{vertical-output: true}

tippe_top = """
<mujoco model="tippe top">
  <option integrator="RK4"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" 
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>

  <worldbody>
    <geom size=".2 .2 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
    <body name="top" pos="0 0 .02">
      <freejoint/>
      <geom name="ball" type="sphere" size=".02" />
      <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008"/>
      <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015" 
       contype="0" conaffinity="0" group="3"/>
    </body>
  </worldbody>

  <keyframe>
    <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0 0 0 0 1 200" />
  </keyframe>
</mujoco>
"""
physics = mujoco.Physics.from_xml_string(tippe_top)
okay = PIL.Image.fromarray(physics.render(camera_id='closeup'))
# okay.show()

#@title Video of the tippe-top {vertical-output: true}

duration = 7    # (seconds)
framerate = 60  # (Hz)

# Simulate and display video.
# frames = []
# physics.reset(0)  # Reset to keyframe 0 (load a saved state).
# while physics.data.time < duration:
#   physics.step()
#   if len(frames) < (physics.data.time) * framerate:
#     pixels = physics.render(camera_id='closeup')
#     frames.append(pixels)
#
# okay = display_video(frames, framerate)
# print(okay)
# plt.show()

