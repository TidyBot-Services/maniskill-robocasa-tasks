"""
Base imports for RoboCasa task classes ported to ManiSkill.

Replaces `from robocasa.environments.kitchen.kitchen import *`
"""

# Re-export the ManiSkill RoboCasaKitchenEnv with evaluate() override
from mani_skill.envs.tasks.mobile_manipulation.robocasa.kitchen import RoboCasaKitchenEnv

# Fixture classes used directly in task files (e.g. `Microwave`, `Stove`)
try:
    from mani_skill.utils.scene_builder.robocasa.fixtures.microwave import Microwave
    from mani_skill.utils.scene_builder.robocasa.fixtures.stove import Stove
    from mani_skill.utils.scene_builder.robocasa.fixtures.sink import Sink
    from mani_skill.utils.scene_builder.robocasa.fixtures.fridge import Fridge
    from mani_skill.utils.scene_builder.robocasa.fixtures.counter import Counter
    from mani_skill.utils.scene_builder.robocasa.fixtures.dishwasher import Dishwasher
except ImportError:
    pass


class _RandomStateShim:
    """Wraps numpy RandomState to expose Generator-like API (integers, choice, etc.)."""
    def __init__(self, rs):
        self._rs = rs

    def __getattr__(self, name):
        return getattr(self._rs, name)

    def integers(self, low, high=None, size=None, endpoint=False):
        if high is None:
            high, low = low, 0
        if endpoint:
            high = high + 1
        return self._rs.randint(low, high, size=size)

    def choice(self, a, size=None, replace=True, p=None, shuffle=True):
        # RandomState.choice only supports 1-D arrays or int
        # Generator.choice supports multi-dimensional. Emulate it:
        if isinstance(a, (list, tuple)) and len(a) > 0 and isinstance(a[0], (list, tuple)):
            idx = self._rs.randint(0, len(a))
            return a[idx]
        try:
            return self._rs.choice(a, size=size, replace=replace, p=p)
        except ValueError:
            # Fallback for empty or weird input
            if len(a) == 0:
                raise
            idx = self._rs.randint(0, len(a))
            return a[idx]

# Fixture types and utilities
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import FixtureType
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder

# Registration
from mani_skill.utils.registration import register_env

# OU utilities (our SAPIEN port)
from robocasa_tasks import robocasa_utils as OU

# Common imports used by task files
import numpy as np


# ---------------------------------------------------------------------------
# Monkey-patch fixture door state methods for SAPIEN compatibility.
# Original methods use MuJoCo API (env.sim.data.qpos); we replace with
# SAPIEN articulation API (self.articulation.get_qpos / set_qpos).
# ---------------------------------------------------------------------------
def _normalize_joint_value(qpos, joint_min, joint_max):
    if joint_max == joint_min:
        return 0.0
    return float(np.clip((qpos - joint_min) / (joint_max - joint_min), 0.0, 1.0))


def _get_joint_qpos(fixture, joint_suffix):
    """Get a single joint's qpos from a SAPIEN articulation."""
    art = getattr(fixture, 'articulation', None)
    if art is None:
        return 0.0
    joint_name = f"{joint_suffix}"
    # Try exact name first, then with fixture name prefix
    for name_candidate in [joint_name, f"{fixture.name}_{joint_suffix}"]:
        if name_candidate in art.active_joints_map:
            idx = art.active_joints_map[name_candidate].active_index[0].item()
            return float(art.get_qpos()[0, idx].cpu().item())
    # Fallback: search by suffix
    for jname, joint in art.active_joints_map.items():
        if joint_suffix in jname:
            idx = joint.active_index[0].item()
            return float(art.get_qpos()[0, idx].cpu().item())
    return 0.0


def _set_joint_qpos(fixture, joint_suffix, value):
    """Set a single joint's qpos on a SAPIEN articulation."""
    art = getattr(fixture, 'articulation', None)
    if art is None:
        return
    for name_candidate in [joint_suffix, f"{fixture.name}_{joint_suffix}"]:
        if name_candidate in art.active_joints_map:
            idx = art.active_joints_map[name_candidate].active_index[0].item()
            qpos = art.get_qpos()
            qpos[:, idx] = value
            art.set_qpos(qpos)
            return
    # Fallback: search by suffix
    for jname, joint in art.active_joints_map.items():
        if joint_suffix in jname:
            idx = joint.active_index[0].item()
            qpos = art.get_qpos()
            qpos[:, idx] = value
            art.set_qpos(qpos)
            return


def _patch_door_state_methods():
    """Patch all fixture classes with SAPIEN-compatible door state methods."""
    from mani_skill.utils.scene_builder.robocasa.fixtures.cabinet import (
        SingleCabinet, HingeCabinet, Drawer,
    )

    # --- SingleCabinet ---
    def _single_set_door_state(self, min, max, env, rng):
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max
        joint_min, joint_max = 0, np.pi / 2
        desired_min = joint_min + (joint_max - joint_min) * min
        desired_max = joint_min + (joint_max - joint_min) * max
        sign = -1 if self.orientation == "left" else 1
        _set_joint_qpos(self, "doorhinge", sign * rng.uniform(desired_min, desired_max))

    def _single_get_door_state(self, env):
        hinge_qpos = _get_joint_qpos(self, "doorhinge")
        sign = -1 if self.orientation == "left" else 1
        hinge_qpos = hinge_qpos * sign
        door = _normalize_joint_value(hinge_qpos, joint_min=0, joint_max=np.pi / 2)
        return {"door": door}

    SingleCabinet.set_door_state = _single_set_door_state
    SingleCabinet.get_door_state = _single_get_door_state

    # --- HingeCabinet ---
    def _hinge_set_door_state(self, min, max, env, rng):
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max
        joint_min, joint_max = 0, np.pi / 2
        desired_min = joint_min + (joint_max - joint_min) * min
        desired_max = joint_min + (joint_max - joint_min) * max
        _set_joint_qpos(self, "rightdoorhinge", rng.uniform(desired_min, desired_max))
        _set_joint_qpos(self, "leftdoorhinge", -rng.uniform(desired_min, desired_max))

    def _hinge_get_door_state(self, env):
        right_qpos = _get_joint_qpos(self, "rightdoorhinge")
        left_qpos = -_get_joint_qpos(self, "leftdoorhinge")
        left_door = _normalize_joint_value(left_qpos, joint_min=0, joint_max=np.pi / 2)
        right_door = _normalize_joint_value(right_qpos, joint_min=0, joint_max=np.pi / 2)
        return {"left_door": left_door, "right_door": right_door}

    HingeCabinet.set_door_state = _hinge_set_door_state
    HingeCabinet.get_door_state = _hinge_get_door_state

    # --- Drawer ---
    def _drawer_set_door_state(self, min, max, env, rng):
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max
        joint_max = self.size[1] * 0.55
        desired_min = joint_max * min
        desired_max = joint_max * max
        _set_joint_qpos(self, "slidejoint", -rng.uniform(desired_min, desired_max))

    def _drawer_get_door_state(self, env):
        qpos = _get_joint_qpos(self, "slidejoint")
        qpos = qpos * -1
        door = _normalize_joint_value(qpos, joint_min=0, joint_max=self.size[1] * 0.55)
        return {"door": door}

    Drawer.set_door_state = _drawer_set_door_state
    Drawer.get_door_state = _drawer_get_door_state

    # --- Microwave ---
    def _micro_set_door_state(self, min, max, env, rng):
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max
        joint_min, joint_max = 0, np.pi / 2
        desired_min = joint_min + (joint_max - joint_min) * min
        desired_max = joint_min + (joint_max - joint_min) * max
        _set_joint_qpos(self, "microjoint", -rng.uniform(desired_min, desired_max))

    def _micro_get_door_state(self, env):
        qpos = _get_joint_qpos(self, "microjoint")
        qpos = -qpos
        door = _normalize_joint_value(qpos, joint_min=0, joint_max=np.pi / 2)
        return {"door": door}

    Microwave.set_door_state = _micro_set_door_state
    Microwave.get_door_state = _micro_get_door_state


_patch_door_state_methods()


# ---------------------------------------------------------------------------
# Monkey-patch Stove methods for SAPIEN compatibility.
# ---------------------------------------------------------------------------
def _patch_stove_methods():
    from mani_skill.utils.scene_builder.robocasa.fixtures.stove import Stove, STOVE_LOCATIONS

    def _stove_get_knobs_state(self, env):
        """SAPIEN-compatible: read knob joint qpos from articulation."""
        knobs_state = {}
        art = getattr(self, 'articulation', None)
        if art is None:
            return knobs_state
        for location in STOVE_LOCATIONS:
            joint = self.knob_joints[location]
            if joint is None:
                continue
            site = self.burner_sites[location]
            if site is None:
                continue
            joint_name = "knob_{}_joint".format(location)
            qpos = _get_joint_qpos(self, joint_name)
            qpos = qpos % (2 * np.pi)
            if qpos < 0:
                qpos += 2 * np.pi
            knobs_state[location] = qpos
        return knobs_state

    def _stove_set_knob_state(self, env, rng, knob, mode="on"):
        """SAPIEN-compatible: set knob joint qpos."""
        assert mode in ["on", "off"]
        if mode == "off":
            joint_val = 0.0
        else:
            if rng.uniform() < 0.5:
                joint_val = rng.uniform(0.50, np.pi / 2)
            else:
                joint_val = rng.uniform(2 * np.pi - np.pi / 2, 2 * np.pi - 0.50)
        joint_name = "knob_{}_joint".format(knob)
        _set_joint_qpos(self, joint_name, joint_val)

    Stove.get_knobs_state = _stove_get_knobs_state
    Stove.set_knob_state = _stove_set_knob_state


_patch_stove_methods()


# ---------------------------------------------------------------------------
# Monkey-patch Sink methods for SAPIEN compatibility.
# ---------------------------------------------------------------------------
def _patch_sink_methods():
    from mani_skill.utils.scene_builder.robocasa.fixtures.sink import Sink

    def _sink_get_handle_state(self, env):
        handle_state = {}
        if self.handle_joint is None:
            return handle_state

        handle_qpos = _get_joint_qpos(self, "handle_joint")
        handle_qpos = handle_qpos % (2 * np.pi)
        if handle_qpos < 0:
            handle_qpos += 2 * np.pi
        handle_state["handle_joint"] = handle_qpos
        handle_state["water_on"] = 0.40 < handle_qpos < np.pi

        spout_qpos = _get_joint_qpos(self, "spout_joint")
        spout_qpos = spout_qpos % (2 * np.pi)
        if spout_qpos < 0:
            spout_qpos += 2 * np.pi
        handle_state["spout_joint"] = spout_qpos
        if np.pi <= spout_qpos <= 2 * np.pi - np.pi / 6:
            spout_ori = "left"
        elif np.pi / 6 <= spout_qpos <= np.pi:
            spout_ori = "right"
        else:
            spout_ori = "center"
        handle_state["spout_ori"] = spout_ori

        return handle_state

    def _sink_set_handle_state(self, env, rng, mode="on"):
        assert mode in ["on", "off", "random"]
        if mode == "random":
            mode = rng.choice(["on", "off"])
        if mode == "off":
            joint_val = 0.0
        elif mode == "on":
            joint_val = rng.uniform(0.40, 0.50)
        _set_joint_qpos(self, "handle_joint", joint_val)

    Sink.get_handle_state = _sink_get_handle_state
    Sink.set_handle_state = _sink_set_handle_state


_patch_sink_methods()


# ---------------------------------------------------------------------------
# Monkey-patch CoffeeMachine methods for SAPIEN compatibility.
# ---------------------------------------------------------------------------
def _patch_coffee_machine_methods():
    from mani_skill.utils.scene_builder.robocasa.fixtures.accessories import CoffeeMachine

    def _coffee_check_receptacle_placement(self, env, obj_name, xy_thresh=0.04):
        if self._receptacle_pouring_site is None:
            return False
        from robocasa_tasks.robocasa_utils import _get_obj_pos
        obj_pos = _get_obj_pos(env, obj_name)
        # Approximate pour site with fixture position (receptacle is below machine)
        pour_pos = np.array(self.pos, dtype=float)
        pour_pos[2] -= 0.05
        xy_check = np.linalg.norm(obj_pos[0:2] - pour_pos[0:2]) < xy_thresh
        z_check = np.abs(obj_pos[2] - pour_pos[2]) < 0.10
        return xy_check and z_check

    def _coffee_gripper_button_far(self, env, th=0.15):
        from robocasa_tasks.robocasa_utils import _get_eef_pos
        gripper_pos = _get_eef_pos(env)
        # Approximate button position with fixture position
        button_pos = np.array(self.pos, dtype=float)
        return float(np.linalg.norm(gripper_pos - button_pos)) > th

    CoffeeMachine.check_receptacle_placement_for_pouring = _coffee_check_receptacle_placement
    CoffeeMachine.gripper_button_far = _coffee_gripper_button_far


_patch_coffee_machine_methods()


# ---------------------------------------------------------------------------
# Monkey-patch Microwave.update_state for SAPIEN compatibility.
# Original uses MuJoCo contact detection for button presses.
# In SAPIEN we skip contact-based toggle and allow manual _turned_on setting.
# ---------------------------------------------------------------------------
def _patch_microwave_state_methods():
    from mani_skill.utils.scene_builder.robocasa.fixtures.microwave import Microwave

    def _micro_update_state(self, env):
        """SAPIEN-compatible: skip MuJoCo contact-based button detection.
        The turned_on state can be set manually or checked via door state."""
        pass

    def _micro_set_turned_on(self, mode, env=None):
        """Convenience method to set microwave on/off."""
        self._turned_on = (mode == "on") if isinstance(mode, str) else bool(mode)

    Microwave.update_state = _micro_update_state
    Microwave.set_turned_on = _micro_set_turned_on


_patch_microwave_state_methods()


class _FixtureRefsProxy:
    """
    Proxy that makes self.fixture_refs behave like a dict (RoboCasa API)
    while the underlying ManiSkill storage is a list-of-dicts.
    """
    def __init__(self, refs_list, idx):
        object.__setattr__(self, '_refs_list', refs_list)
        object.__setattr__(self, '_idx', idx)

    def _ensure_idx(self, idx):
        refs_list = object.__getattribute__(self, '_refs_list')
        while len(refs_list) <= idx:
            refs_list.append({})

    def __getitem__(self, key):
        if isinstance(key, int):
            refs_list = object.__getattribute__(self, '_refs_list')
            return refs_list[key]
        idx = object.__getattribute__(self, '_idx')
        self._ensure_idx(idx)
        refs_list = object.__getattribute__(self, '_refs_list')
        return refs_list[idx][key]

    def __setitem__(self, key, value):
        if isinstance(key, int):
            refs_list = object.__getattribute__(self, '_refs_list')
            refs_list[key] = value
            return
        idx = object.__getattribute__(self, '_idx')
        self._ensure_idx(idx)
        refs_list = object.__getattribute__(self, '_refs_list')
        refs_list[idx][key] = value

    def __contains__(self, key):
        if isinstance(key, int):
            refs_list = object.__getattribute__(self, '_refs_list')
            return key < len(refs_list)
        idx = object.__getattribute__(self, '_idx')
        self._ensure_idx(idx)
        refs_list = object.__getattribute__(self, '_refs_list')
        return key in refs_list[idx]

    def __iter__(self):
        idx = object.__getattribute__(self, '_idx')
        refs_list = object.__getattribute__(self, '_refs_list')
        if idx < len(refs_list):
            return iter(refs_list[idx])
        return iter({})

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def items(self):
        idx = object.__getattribute__(self, '_idx')
        refs_list = object.__getattribute__(self, '_refs_list')
        if idx < len(refs_list):
            return refs_list[idx].items()
        return {}.items()

    def keys(self):
        idx = object.__getattribute__(self, '_idx')
        refs_list = object.__getattribute__(self, '_refs_list')
        if idx < len(refs_list):
            return refs_list[idx].keys()
        return {}.keys()

    def values(self):
        idx = object.__getattribute__(self, '_idx')
        refs_list = object.__getattribute__(self, '_refs_list')
        if idx < len(refs_list):
            return refs_list[idx].values()
        return {}.values()

    def append(self, item):
        """Support list-style append (ManiSkill uses fixture_refs.append({}))."""
        refs_list = object.__getattribute__(self, '_refs_list')
        refs_list.append(item)
        # Update idx to point to newly appended item
        object.__setattr__(self, '_idx', len(refs_list) - 1)

    def __len__(self):
        refs_list = object.__getattribute__(self, '_refs_list')
        return len(refs_list)


_FLAT_OBJECT_KEYWORDS = ("container", "plate", "bowl", "cup", "pan", "pot", "tray", "mug", "kettle", "cutting_board")
_TILT_THRESHOLD_RAD = np.deg2rad(15.0)
_FLOOR_Z_THRESHOLD = 0.5   # objects below this z likely fell off a surface
_COUNTER_Z_DEFAULT = 0.93  # typical kitchen counter height in RoboCasa scenes
_FLOAT_Z_MARGIN = 0.15     # objects more than this above surface_z are floating


class Kitchen(RoboCasaKitchenEnv):
    """
    Base class for all ported RoboCasa tasks.

    Overrides evaluate() to call _check_success() and return
    {"success": bool} for ManiSkill compatibility.
    """

    def _reset_internal(self):
        """Override point for task subclasses to set initial fixture state (door open, knob on, etc).
        Called from _load_scene after scene is built. Subclasses should call super()._reset_internal()."""
        pass

    def _initialize_episode(self, env_idx, options):
        """Parent init + correct physics issues (tilted/fallen objects)."""
        super()._initialize_episode(env_idx, options)
        self._stabilize_scene_objects()

    def _recompute_robot_base_pose(self):
        """Recompute robot base pose using init_robot_base_pos after _setup_kitchen_references."""
        import torch
        from transforms3d.euler import euler2quat

        ref_fixture = getattr(self, 'init_robot_base_pos', None)
        sb = getattr(self, 'scene_builder', None)
        if ref_fixture is None or sb is None or sb.robot_poses is None:
            return

        for scene_idx in range(self.num_envs):
            fixtures = sb.scene_data[scene_idx].get("fixtures", {})
            try:
                robot_base_pos, robot_base_ori = sb.compute_robot_base_placement_pose(
                    fixtures, ref_fixture
                )
                sb.robot_poses.raw_pose[scene_idx][:3] = torch.from_numpy(
                    robot_base_pos
                ).to(sb.robot_poses.device)
                sb.robot_poses.raw_pose[scene_idx][3:] = torch.from_numpy(
                    euler2quat(*robot_base_ori)
                ).to(sb.robot_poses.device)
            except Exception:
                pass  # Fall back to original pose if recomputation fails

    def stabilize_scene(self):
        """Public method: re-stabilize objects after physics steps."""
        self._stabilize_scene_objects()

    def _stabilize_scene_objects(self):
        """Fix common physics issues: tilted flat objects + fallen objects."""
        from scipy.spatial.transform import Rotation as R
        import sapien
        import torch

        for scene_idx in range(self.num_envs):
            # Collect all object z-positions to estimate surface height
            all_z = []
            for data in self.object_actors[scene_idx].values():
                z = data["actor"].pose.p[0, 2].item()
                if z > _FLOOR_Z_THRESHOLD:
                    all_z.append(z)
            surface_z = np.median(all_z) if all_z else _COUNTER_Z_DEFAULT

            for name, data in self.object_actors[scene_idx].items():
                actor = data["actor"]
                pose = actor.pose
                pos = pose.p[0].cpu().numpy()
                quat_wxyz = pose.q[0].cpu().numpy()
                quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
                euler = R.from_quat(quat_xyzw).as_euler('xyz')
                changed = False

                # 1. Fix fallen objects: teleport back to surface height
                if pos[2] < _FLOOR_Z_THRESHOLD:
                    init_pose = data.get("pose")
                    if init_pose is not None:
                        # Restore to original placement pose
                        actor.set_pose(init_pose)
                        pos = init_pose.p[0].cpu().numpy() if hasattr(init_pose.p, 'cpu') else np.array(init_pose.p)
                        qw = init_pose.q[0].cpu().numpy() if hasattr(init_pose.q, 'cpu') else np.array(init_pose.q)
                        quat_xyzw = [qw[1], qw[2], qw[3], qw[0]]
                        euler = R.from_quat(quat_xyzw).as_euler('xyz')
                    else:
                        # No saved pose — place on counter at original xy
                        pos[2] = surface_z + 0.02
                    changed = True

                # 2. Fix tilted flat objects (plates, bowls, containers, etc.)
                is_flat_type = any(kw in name.lower() for kw in _FLAT_OBJECT_KEYWORDS)
                tilt = max(abs(euler[0]), abs(euler[1]))
                if is_flat_type and tilt > _TILT_THRESHOLD_RAD:
                    corrected = R.from_euler('xyz', [0.0, 0.0, euler[2]])
                    cq = corrected.as_quat()  # xyzw
                    new_quat = np.array([cq[3], cq[0], cq[1], cq[2]])  # wxyz
                    # If severely tilted (>60°), z needs correction
                    if tilt > np.deg2rad(60.0) and pos[2] > surface_z + 0.05:
                        pos[2] = surface_z + 0.02
                    new_pose = sapien.Pose(pos, new_quat)
                    actor.set_pose(new_pose)
                    data["pose"] = new_pose
                    changed = True
                elif not is_flat_type and tilt > np.deg2rad(25.0):
                    # Any object significantly tilted (>25°) — upright it
                    corrected = R.from_euler('xyz', [0.0, 0.0, euler[2]])
                    cq = corrected.as_quat()
                    new_quat = np.array([cq[3], cq[0], cq[1], cq[2]])
                    new_pose = sapien.Pose(pos, new_quat)
                    actor.set_pose(new_pose)
                    data["pose"] = new_pose
                    changed = True

                # 3. Fix floating objects (bounced up during physics)
                #    Compare current z to initial placement z
                init_pose = data.get("pose")
                if init_pose is not None:
                    init_z = init_pose.p[0, 2].item() if hasattr(init_pose.p, 'cpu') else init_pose.p[2]
                    drift_z = pos[2] - init_z
                    if drift_z > _FLOAT_Z_MARGIN:
                        pos[2] = init_z
                        cur_q = actor.pose.q[0].cpu().numpy()
                        new_pose = sapien.Pose(pos, cur_q)
                        actor.set_pose(new_pose)
                        data["pose"] = new_pose
                        changed = True

                # 4. Fix partial penetration: object center on counter but
                #    collision mesh extends below surface
                if 0.85 < pos[2] < 1.05:
                    try:
                        body = actor._bodies[0]
                        aabb = body.compute_global_aabb_tight()
                        aabb_bottom = aabb[0][2]
                        if aabb_bottom < surface_z - 0.01:
                            lift = (surface_z - aabb_bottom) + 0.002
                            pos[2] += lift
                            cur_q = actor.pose.q[0].cpu().numpy()
                            new_pose = sapien.Pose(pos, cur_q)
                            actor.set_pose(new_pose)
                            data["pose"] = new_pose
                            changed = True
                    except Exception:
                        pass

                if changed:
                    actor.set_linear_velocity(torch.zeros(1, 3, device=self.device))
                    actor.set_angular_velocity(torch.zeros(1, 3, device=self.device))

    @property
    def rng(self):
        """Compatibility shim: RoboCasa uses self.rng, ManiSkill uses _batched_episode_rng.
        Returns a numpy Generator (has .integers(), .choice(), etc.)"""
        if hasattr(self, '_batched_episode_rng') and self._batched_episode_rng is not None:
            rng = self._batched_episode_rng[0]
            # _batched_episode_rng may be a numpy Generator already
            if hasattr(rng, 'integers'):
                return rng
            # Wrap RandomState in a Generator-compatible shim
            return _RandomStateShim(rng)
        import numpy as np
        return np.random.default_rng(0)
    
    def evaluate(self):
        import torch
        if hasattr(self, '_check_success'):
            # Lazy-init attributes that _reset_internal would have set
            self._ensure_reset_attrs()
            try:
                success = self._check_success()
                return {"success": torch.tensor([bool(success)], dtype=torch.bool, device=self.device)}
            except Exception as e:
                return {"success": torch.tensor([False], dtype=torch.bool, device=self.device), "error": str(e)}
        return {"success": torch.tensor([False], dtype=torch.bool, device=self.device)}

    def compute_robot_base_placement_pose(self, fixture=None, ref_fixture=None, offset=None, **kwargs):
        """Compute robot base placement: stand in front of fixture, facing it.

        Args:
            fixture: The fixture to stand in front of.
            ref_fixture: Alternative fixture reference (takes priority over fixture).
            offset: (x, y) offset in the fixture's local frame.
                    Local frame: X = left/right, Y-negative = in front.

        Returns:
            (pos_2d, yaw): Robot base XY position and yaw angle facing the fixture.
        """
        from transforms3d.euler import euler2mat

        fxtr = ref_fixture if ref_fixture is not None else fixture
        if fxtr is None:
            return np.array([0.5, 0.0]), 0.0

        fxtr_pos = np.array(getattr(fxtr, 'pos', [0, 0, 0]), dtype=float)
        fxtr_rot = float(getattr(fxtr, 'rot', 0.0))
        rot_mat = euler2mat(0, 0, fxtr_rot)

        if offset is not None:
            offset_3d = np.array([offset[0], offset[1], 0.0])
        else:
            # Default: stand 0.5m in front of fixture (local -Y direction)
            offset_3d = np.array([0.0, -0.5, 0.0])

        world_pos = fxtr_pos + rot_mat @ offset_3d
        robot_pos = world_pos[:2]

        # Yaw: face toward the fixture
        dx = fxtr_pos[0] - robot_pos[0]
        dy = fxtr_pos[1] - robot_pos[1]
        yaw = float(np.arctan2(dy, dx))

        return robot_pos, yaw

    def _ensure_reset_attrs(self):
        """Set attributes that _reset_internal() normally sets, using SAPIEN state."""
        # self.knob — used by stove tasks (SearingMeat, MultistepSteaming, etc.)
        if not hasattr(self, 'knob') and hasattr(self, 'stove') and hasattr(self, 'knob_id'):
            try:
                valid = list(self.stove.get_knobs_state(env=self).keys())
                if valid:
                    self.knob = valid[0]  # default to first knob
            except Exception:
                pass
        # self.target_location — used by some navigate tasks
        if not hasattr(self, 'target_location') and hasattr(self, 'init_robot_base_pos'):
            self.target_location = getattr(self, 'init_robot_base_pos', None)
    
    def get_obj_lang(self):
        """Get language description of the main object."""
        scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
        if "obj" in self.object_actors[scene_idx]:
            return "object"
        return "item"
    
    def check_contact(self, obj1, obj2):
        """Stub for MuJoCo contact check — use proximity instead."""
        # In SAPIEN, we approximate contact with distance
        try:
            scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
            
            # Get positions
            def get_pos(obj):
                if hasattr(obj, 'pose'):
                    return obj.pose.p[0].cpu().numpy()
                if hasattr(obj, 'pos'):
                    return np.array(obj.pos)
                # Try as object name
                if isinstance(obj, str):
                    if obj in self.object_actors[scene_idx]:
                        return self.object_actors[scene_idx][obj]["actor"].pose.p[0].cpu().numpy()
                # Try name attribute
                if hasattr(obj, 'name'):
                    name = obj.name
                    if name in self.object_actors[scene_idx]:
                        return self.object_actors[scene_idx][name]["actor"].pose.p[0].cpu().numpy()
                return None
            
            p1 = get_pos(obj1)
            p2 = get_pos(obj2)
            if p1 is not None and p2 is not None:
                return float(np.linalg.norm(p1 - p2)) < 0.15
        except Exception:
            pass
        return False
    
    def __getattribute__(self, name):
        if name in ('fixture_refs', 'objects'):
            refs_list = object.__getattribute__(self, name)
            if isinstance(refs_list, _FixtureRefsProxy):
                return refs_list
            if isinstance(refs_list, list):
                idx = object.__getattribute__(self, '_scene_idx_to_be_loaded') if hasattr(self, '_scene_idx_to_be_loaded') else 0
                return _FixtureRefsProxy(refs_list, idx)
            return refs_list
        return object.__getattribute__(self, name)

    @property
    def fixtures(self):
        """RoboCasa-compatible access to scene fixtures dict."""
        idx = getattr(self, '_scene_idx_to_be_loaded', 0)
        if hasattr(self, 'scene_builder') and hasattr(self.scene_builder, 'scene_data'):
            if idx < len(self.scene_builder.scene_data):
                return self.scene_builder.scene_data[idx].get("fixtures", {})
        return {}

    def _load_scene(self, options):
        """Ensure _setup_kitchen_references is always called, even for tasks without _get_obj_cfgs."""
        # Check if parent would skip _setup_kitchen_references
        if not self.fixtures_only and not hasattr(self, '_get_obj_cfgs'):
            # Provide a minimal _get_obj_cfgs so parent runs _setup_kitchen_references
            self._get_obj_cfgs = lambda: []
        result = super()._load_scene(options)

        # Recompute robot base pose: the scene builder computes robot position
        # during build() before _setup_kitchen_references() sets init_robot_base_pos,
        # so the robot ends up at a random fixture. Fix by recomputing after task
        # has set init_robot_base_pos.
        self._recompute_robot_base_pose()

        # Try to run _reset_internal to set task-specific attrs (e.g. door state, knobs)
        # Guard: only if defined on a task subclass (not on Kitchen/RoboCasaKitchenEnv base)
        has_custom_reset = any(
            '_reset_internal' in cls.__dict__
            for cls in type(self).__mro__
            if cls is not Kitchen and cls is not RoboCasaKitchenEnv
            and issubclass(cls, Kitchen)
        )
        if has_custom_reset:
            try:
                self._reset_internal()
            except Exception:
                pass  # Silently ignore SAPIEN-incompatible parts
        return result

    def register_fixture_ref(self, ref_name, fn_kwargs):
        """Override to handle missing fixture types gracefully."""
        try:
            return super().register_fixture_ref(ref_name, fn_kwargs)
        except (KeyError, AssertionError):
            # Fixture type not present in this scene layout — try fallback to COUNTER
            scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
            raw_refs = object.__getattribute__(self, 'fixture_refs')
            try:
                from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import FixtureType
                fallback = self.scene_builder.get_fixture(
                    self.scene_builder.scene_data[scene_idx]["fixtures"],
                    id=FixtureType.COUNTER
                )
                raw_refs[scene_idx][ref_name] = fallback
                return fallback
            except Exception:
                raw_refs[scene_idx][ref_name] = None
                return None

    def get_fixture(self, fixture_id=None, ref=None, ref_fixture=None, id=None, size=None):
        """Get a fixture by id (compatibility with RoboCasa API).
        
        Args:
            fixture_id: FixtureType or string id (positional)
            id: same as fixture_id (keyword form used by some tasks)
            ref: optional reference fixture for proximity search
            ref_fixture: alias for ref
            size: minimum size for counter fixtures
        """
        if id is not None and fixture_id is None:
            fixture_id = id
        scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
        # Delegate to scene builder if possible
        if hasattr(self, 'scene_builder') and self.scene_builder is not None:
            fixtures = self.scene_builder.scene_data[scene_idx]["fixtures"]
            try:
                return self.scene_builder.get_fixture(fixtures, fixture_id,
                                                       ref=ref or ref_fixture)
            except Exception:
                pass
        # Fallback: search fixture_refs
        if isinstance(fixture_id, str) and fixture_id in self.fixture_refs[scene_idx]:
            return self.fixture_refs[scene_idx][fixture_id]
        return None

    @property
    def sim(self):
        """Stub: RoboCasa tasks call self.sim (MuJoCo). Return env-aware shim."""
        return _SimShim(self)

    @property
    def obj_body_id(self):
        """RoboCasa compat: maps object name → integer index (used for body_xpos lookup)."""
        scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
        mapping = {}
        for i, (name, info) in enumerate(self.object_actors[scene_idx].items()):
            mapping[name] = i
            # Also map via the object's .name attribute if different
            actor = info.get("actor")
            if actor and hasattr(actor, 'name') and actor.name != name:
                mapping[actor.name] = i
        return mapping

    def _get_objects_dict(self):
        """RoboCasa compat: returns object dict for current env index."""
        scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
        return {k: v["actor"] for k, v in self.object_actors[scene_idx].items()}

    def _get_fixtures_dict(self):
        """RoboCasa compat: returns fixture dict for current env index."""
        scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
        return self.fixture_refs[scene_idx]


class _SimShim:
    """Env-aware shim for self.sim calls from RoboCasa _check_success methods."""
    model = None  # No MuJoCo model

    def __init__(self, env=None):
        self._env = env
        self.data = _SimData(env)

    def __getattr__(self, name):
        return None


class _SimData:
    """Shim for sim.data — provides body_xpos and get_site_xpos via SAPIEN."""

    def __init__(self, env):
        self._env = env

    @property
    def body_xpos(self):
        """Returns a dict-like object: body_xpos[idx] → position array."""
        return _BodyXposProxy(self._env)

    def get_site_xpos(self, site_name):
        """Look up site position from articulation XML data."""
        env = self._env
        if env is None:
            return np.zeros(3)
        # Search all articulations for a site matching this name
        for art in env.scene.get_all_articulations():
            loader = getattr(art, 'loader', None)
            if loader is None:
                continue
            xml = getattr(loader, 'xml', None)
            if xml is None:
                continue
            site = xml.find(f".//*site[@name='{site_name}']")
            if site is not None:
                pos_str = site.get('pos', '0 0 0')
                local_pos = np.array([float(x) for x in pos_str.split()])
                # Transform by articulation root pose
                art_pos = art.pose.p
                if hasattr(art_pos, 'cpu'):
                    art_pos = art_pos[0].cpu().numpy()
                return np.array(art_pos) + local_pos
        # Fallback: search fixtures
        scene_idx = getattr(env, '_scene_idx_to_be_loaded', 0)
        try:
            for fx_name, fx in env.fixture_refs[scene_idx].items():
                if hasattr(fx, 'loader') and fx.loader is not None:
                    xml = getattr(fx.loader, 'xml', None)
                    if xml is None:
                        continue
                    site = xml.find(f".//*site[@name='{site_name}']")
                    if site is not None:
                        pos_str = site.get('pos', '0 0 0')
                        local_pos = np.array([float(x) for x in pos_str.split()])
                        fx_pos = np.array(getattr(fx, 'pos', [0, 0, 0]))
                        return fx_pos + local_pos
        except Exception:
            pass
        return np.zeros(3)

    @property
    def qpos(self):
        """Allow sim.data.qpos[joint_id] — return zeros (joints not tracked this way)."""
        return _QposProxy(self._env)

    def set_joint_qpos(self, *args, **kwargs):
        pass  # no-op in eval mode

    @property
    def xquat(self):
        """sim.data.xquat[idx] → identity quaternion proxy."""
        return _QuatProxy(self._env)

    @property
    def site_xpos(self):
        """sim.data.site_xpos[site_id] → zeros proxy (site ids not tracked)."""
        return _ZerosProxy()

    def __getattr__(self, name):
        return _ZerosProxy()  # Safe fallback: returns zeros for any attribute access


class _ZerosProxy:
    """Safe fallback proxy that returns np.zeros(3) for any index."""
    def __getitem__(self, idx):
        return np.zeros(3)
    def __call__(self, *a, **kw):
        return np.zeros(3)


class _QuatProxy:
    """sim.data.xquat[idx] → identity quaternion [1,0,0,0]."""
    def __init__(self, env):
        self._env = env

    def __getitem__(self, idx):
        import numpy as _np
        # Try to get actual orientation from actor
        env = self._env
        if env is not None:
            scene_idx = getattr(env, '_scene_idx_to_be_loaded', 0)
            actors = list(env.object_actors[scene_idx].values())
            if isinstance(idx, int) and 0 <= idx < len(actors):
                actor = actors[idx]["actor"]
                q = actor.pose.q
                if hasattr(q, 'cpu'):
                    return q[0].cpu().numpy()
                return _np.array(q)
        return _np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion


class _BodyXposProxy:
    """body_xpos[idx] → object position from SAPIEN actors."""
    def __init__(self, env):
        self._env = env

    def __getitem__(self, idx):
        env = self._env
        if env is None:
            return np.zeros(3)
        scene_idx = getattr(env, '_scene_idx_to_be_loaded', 0)
        actors = list(env.object_actors[scene_idx].values())
        if isinstance(idx, int) and 0 <= idx < len(actors):
            actor = actors[idx]["actor"]
            p = actor.pose.p
            if hasattr(p, 'cpu'):
                return p[0].cpu().numpy()
            return np.array(p)
        return np.zeros(3)


class _QposProxy:
    """sim.data.qpos[joint_id] → 0.0 fallback."""
    def __init__(self, env):
        self._env = env

    def __getitem__(self, idx):
        return 0.0
