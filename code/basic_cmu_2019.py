from dm_control import composer
from dm_control.locomotion import arenas

from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.tasks.reference_pose import tracking

from dm_control.locomotion.walkers import cmu_humanoid


def cmu_humanoid_tracking(random_state=None):
  """Requires a CMU humanoid to run down a corridor obstructed by walls."""

  # Use a position-controlled CMU humanoid walker.
  walker_type = cmu_humanoid.CMUHumanoidPositionControlledV2020

  # Build an empty arena.
  arena = arenas.Floor()

  # Build a task that rewards the agent for tracking motion capture reference
  # data.
  task = tracking.MultiClipMocapTracking(
      walker=walker_type,
      arena=arena,
      ref_path=cmu_mocap_data.get_path_for_cmu(version='2020'),
      dataset='walk_tiny',
      ref_steps=(1, 2, 3, 4, 5),
      min_steps=10,
      reward_type='comic',
  )

  return composer.Environment(time_limit=30,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True)