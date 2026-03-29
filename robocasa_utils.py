"""
RoboCasa success-check utility functions ported to ManiSkill/SAPIEN.

Original: robocasa/utils/object_utils.py (MuJoCo API)
Ported:   SAPIEN actor/contact API

These functions are used by _check_success() in RoboCasa task classes.
"""

import numpy as np


def _get_obj_actor(env, obj_name):
    """Get SAPIEN actor for an object (by name string or actor object)."""
    # If already an actor/object, return its position-provider
    if not isinstance(obj_name, str):
        return obj_name  # assume it's already an actor

    scene_idx = getattr(env, '_scene_idx_to_be_loaded', 0)
    obj_data = env.object_actors[scene_idx].get(obj_name)
    if obj_data is None:
        for k, v in env.object_actors[scene_idx].items():
            if obj_name in k:
                obj_data = v
                break
    if obj_data is None:
        raise KeyError(f"Object '{obj_name}' not found in object_actors")
    return obj_data["actor"]


def _get_obj_pos(env, obj_name) -> np.ndarray:
    """Get object world position from env.object_actors."""
    actor = _get_obj_actor(env, obj_name)
    if hasattr(actor, 'pose'):
        return actor.pose.p[0].cpu().numpy()
    # Fixture-like object with .pos attribute
    if hasattr(actor, 'pos'):
        return np.array(actor.pos)
    raise KeyError(f"Cannot get position for '{obj_name}'")


def _get_eef_pos(env) -> np.ndarray:
    """Get end-effector (EEF) world position."""
    robot = env.agent.robot
    eef_link = robot.links_map.get("eef")
    if eef_link is None:
        # Fallback: try panda_hand or similar
        for name in ["panda_hand", "eef", "gripper_center"]:
            if name in robot.links_map:
                eef_link = robot.links_map[name]
                break
    if eef_link is None:
        raise RuntimeError("Cannot find EEF link on robot")
    return eef_link.pose.p[0].cpu().numpy()


def _get_fixture_ref(env, fixture_name):
    """Get a fixture reference by name or return the fixture object directly."""
    # If it's already a fixture object, return it
    if not isinstance(fixture_name, str):
        return fixture_name
    
    scene_idx = getattr(env, '_scene_idx_to_be_loaded', 0)
    refs = env.fixture_refs[scene_idx]
    if fixture_name in refs:
        return refs[fixture_name]
    # Try partial match
    for k, v in refs.items():
        if isinstance(k, str) and fixture_name in k:
            return v
    raise KeyError(f"Fixture '{fixture_name}' not found in fixture_refs")


# ---------------------------------------------------------------------------
# OU.gripper_obj_far
# ---------------------------------------------------------------------------

def gripper_obj_far(env, obj_name: str = "obj", th: float = 0.25) -> bool:
    """Check if gripper is far from object (distance > threshold).

    Original: robocasa/utils/object_utils.py::gripper_obj_far
    """
    obj_pos = _get_obj_pos(env, obj_name)
    eef_pos = _get_eef_pos(env)
    return float(np.linalg.norm(eef_pos - obj_pos)) > th


# ---------------------------------------------------------------------------
# OU.check_obj_in_receptacle
# ---------------------------------------------------------------------------

def check_obj_in_receptacle(env, obj_name: str, receptacle_name: str,
                            th: float = None) -> bool:
    """Check if object is in/on a receptacle object.

    Original checks contact + XY proximity. We use XY + Z proximity
    since SAPIEN contact queries can be unreliable for resting objects.

    Args:
        env: Unwrapped ManiSkill env
        obj_name: Name of the object
        receptacle_name: Name of the receptacle object
        th: XY distance threshold. If None, uses 0.10m default.
    """
    obj_pos = _get_obj_pos(env, obj_name)
    recep_pos = _get_obj_pos(env, receptacle_name)

    if th is None:
        th = 0.10  # tightened from 0.15

    xy_dist = np.linalg.norm(obj_pos[:2] - recep_pos[:2])
    # Object must be above receptacle (not below or floating far above)
    z_diff = obj_pos[2] - recep_pos[2]
    z_ok = -0.02 <= z_diff <= 0.15  # at most 2cm below, 15cm above

    return bool(xy_dist < th and z_ok)


def check_obj_near_obj(env, obj_name1: str, obj_name2: str, th: float = 0.25) -> bool:
    """Check if two objects are near each other (XY distance within threshold).

    Args:
        env: Unwrapped ManiSkill env
        obj_name1: Name of the first object
        obj_name2: Name of the second object
        th: XY distance threshold in meters
    """
    pos1 = _get_obj_pos(env, obj_name1)
    pos2 = _get_obj_pos(env, obj_name2)
    xy_dist = np.linalg.norm(pos1[:2] - pos2[:2])
    return bool(xy_dist < th)


def check_obj_near_fixture(env, obj_name: str, fixture, th: float = 0.35) -> bool:
    """Check if an object is near a fixture (XY distance within threshold).

    Args:
        env: Unwrapped ManiSkill env
        obj_name: Name of the object
        fixture: Fixture reference
        th: XY distance threshold in meters
    """
    obj_pos = _get_obj_pos(env, obj_name)
    fixture_pos = np.array(fixture.pos)
    xy_dist = np.linalg.norm(obj_pos[:2] - fixture_pos[:2])
    return bool(xy_dist < th)


# ---------------------------------------------------------------------------
# OU.check_obj_location_on_stove
# ---------------------------------------------------------------------------

def check_obj_location_on_stove(env, obj_name, stove, threshold=0.08):
    """Check if object is on the stove near a burner with the knob turned on.

    SAPIEN-compatible replacement for the MuJoCo-based
    _check_obj_location_on_stove method used by multiple tasks.

    Args:
        env: Unwrapped ManiSkill env
        obj_name: Object name in object_actors
        stove: Stove fixture reference
        threshold: Max XY distance to burner center

    Returns:
        str or None: Burner location name (e.g. "front_left") if object is on
                     that burner and the knob is on, else None.
    """
    from transforms3d.euler import euler2mat

    obj_pos = _get_obj_pos(env, obj_name)
    obj_on_stove = check_obj_fixture_contact(env, obj_name, stove)
    if not obj_on_stove:
        return None

    knobs_state = stove.get_knobs_state(env=env)
    stove_pos = np.array(stove.pos)
    stove_rot = float(getattr(stove, 'rot', 0.0))
    rot_mat = euler2mat(0, 0, stove_rot)

    # Get burner positions from reset regions (local offsets)
    regions = stove.get_reset_regions(env=env) if hasattr(stove, 'get_reset_regions') else {}

    for location, region in regions.items():
        if location not in knobs_state:
            continue
        offset = np.array(region.get("offset", [0, 0, 0]), dtype=float)
        burner_world = stove_pos + rot_mat @ offset
        dist = np.linalg.norm(burner_world[:2] - obj_pos[:2])

        if dist < threshold:
            knob_value = knobs_state[location]
            knob_on = 0.35 <= np.abs(knob_value) <= 2 * np.pi - 0.35
            if knob_on:
                return location

    return None


# ---------------------------------------------------------------------------
# OU.obj_inside_of
# ---------------------------------------------------------------------------

def obj_inside_of(env, obj_name: str, fixture_id: str,
                  partial_check: bool = False, th: float = 0.05) -> bool:
    """Check if object is inside a fixture (cabinet, fridge, etc).

    Simplified version: checks if object position is within the fixture's
    bounding region. Uses fixture's int_sites if available, otherwise
    falls back to position + size check.

    Args:
        env: Unwrapped ManiSkill env
        obj_name: Object name in object_actors
        fixture_id: Fixture reference name in fixture_refs
        partial_check: If True, only check object center (not bbox)
        th: Tolerance threshold
    """
    obj_pos = _get_obj_pos(env, obj_name)
    fixture = _get_fixture_ref(env, fixture_id)

    # Try to use fixture's int_sites (interior regions)
    if hasattr(fixture, 'get_int_sites'):
        try:
            int_regions = fixture.get_int_sites(relative=False)
            for (p0, px, py, pz) in int_regions.values():
                p0, px, py, pz = np.array(p0), np.array(px), np.array(py), np.array(pz)
                u = px - p0
                v = py - p0
                w = pz - p0

                # Check if object center is within the region
                check1 = np.dot(u, p0) - th <= np.dot(u, obj_pos) <= np.dot(u, px) + th
                check2 = np.dot(v, p0) - th <= np.dot(v, obj_pos) <= np.dot(v, py) + th
                check3 = np.dot(w, p0) - th <= np.dot(w, obj_pos) <= np.dot(w, pz) + th

                if check1 and check2 and check3:
                    return True
            return False
        except Exception:
            pass

    # Fallback: use fixture position + size, but only allow half-size in Z
    # to avoid including the space below the fixture (e.g., counter below cabinet)
    if hasattr(fixture, 'pos') and hasattr(fixture, 'size'):
        fpos = np.array(fixture.pos)
        fsize = np.array(fixture.size) if hasattr(fixture, 'size') else np.array([0.3, 0.3, 0.3])
        lower = fpos - fsize - th
        upper = fpos + fsize + th
        # Tighten Z: only allow objects within the upper half of the fixture
        # (prevents counting objects on counters below wall cabinets)
        lower[2] = fpos[2] - fsize[2] * 0.5 - th
        return bool(np.all(obj_pos >= lower) and np.all(obj_pos <= upper))

    # Last resort: tightened distance check
    if hasattr(fixture, 'pos'):
        dist = np.linalg.norm(obj_pos - np.array(fixture.pos))
        return dist < 0.25  # tightened from 0.5

    return False


# ---------------------------------------------------------------------------
# OU.check_obj_fixture_contact
# ---------------------------------------------------------------------------

def check_obj_fixture_contact(env, obj_name, fixture_name=None) -> bool:
    """Check if object is in contact with (resting on) a fixture.

    Original uses MuJoCo contact array. We approximate with:
    1. XY within fixture bounding box (using ext_sites if available)
    2. Z height: object must be near fixture surface (not floating or below)

    Args:
        env: Unwrapped ManiSkill env
        obj_name: Object name (str) or object itself
        fixture_name: Fixture reference name/object. If None, returns False.
    """
    if fixture_name is None:
        return False

    obj_pos = _get_obj_pos(env, obj_name if isinstance(obj_name, str) else "obj")
    fixture = _get_fixture_ref(env, fixture_name)

    if fixture is None or not hasattr(fixture, 'pos'):
        return False

    fpos = np.array(fixture.pos)

    # Try bounding box check via ext_sites
    if hasattr(fixture, 'get_ext_sites'):
        try:
            ext = fixture.get_ext_sites(relative=False)
            if len(ext) >= 2:
                # ext_sites gives corner points; check XY containment
                ext_arr = np.array(ext)
                xy_min = ext_arr[:, :2].min(axis=0) - 0.03  # 3cm tolerance
                xy_max = ext_arr[:, :2].max(axis=0) + 0.03
                z_surface = ext_arr[:, 2].max()  # top surface

                xy_in = np.all(obj_pos[:2] >= xy_min) and np.all(obj_pos[:2] <= xy_max)
                # Object should be near the surface (within 10cm above, 2cm below)
                z_ok = -0.02 <= (obj_pos[2] - z_surface) <= 0.10
                return bool(xy_in and z_ok)
        except Exception:
            pass

    # Fallback: use fixture position + size for bounding box
    if hasattr(fixture, 'size'):
        fsize = np.array(fixture.size)
        # XY: within fixture footprint + small tolerance
        xy_ok = (np.abs(obj_pos[0] - fpos[0]) < fsize[0] + 0.03 and
                 np.abs(obj_pos[1] - fpos[1]) < fsize[1] + 0.03)
        # Z: object should be near fixture top surface
        fixture_top = fpos[2] + fsize[2]
        z_ok = -0.02 <= (obj_pos[2] - fixture_top) <= 0.10
        return bool(xy_ok and z_ok)

    # Last resort: tightened distance check
    dist = np.linalg.norm(obj_pos - fpos)
    return dist < 0.10  # tightened from 0.15


# ---------------------------------------------------------------------------
# Additional utilities used by some tasks
# ---------------------------------------------------------------------------

def gripper_fxtr_far(env, fixture_name: str, th: float = 0.25) -> bool:
    """Check if gripper is far from a fixture."""
    eef_pos = _get_eef_pos(env)
    fixture = _get_fixture_ref(env, fixture_name)
    if hasattr(fixture, 'pos'):
        fpos = np.array(fixture.pos)
        return float(np.linalg.norm(eef_pos - fpos)) > th
    return True


def fixture_pairwise_dist(f1, f2) -> float:
    """Distance between two fixtures using their exterior bounding box points."""
    if hasattr(f1, 'get_ext_sites') and hasattr(f2, 'get_ext_sites'):
        f1_points = f1.get_ext_sites(all_points=True, relative=False)
        f2_points = f2.get_ext_sites(all_points=True, relative=False)
        all_dists = [np.linalg.norm(np.array(p1) - np.array(p2)) for p1 in f1_points for p2 in f2_points]
        return float(np.min(all_dists))
    # Fallback: use positions
    p1 = np.array(f1.pos) if hasattr(f1, 'pos') else np.zeros(3)
    p2 = np.array(f2.pos) if hasattr(f2, 'pos') else np.zeros(3)
    return float(np.linalg.norm(p1 - p2))


def point_in_fixture(env, point, fixture, th=0.05) -> bool:
    """Check if a 3D point is inside a fixture's bounding region."""
    fixture = _get_fixture_ref(env, fixture)
    point = np.array(point)
    
    if hasattr(fixture, 'get_int_sites'):
        try:
            int_regions = fixture.get_int_sites(relative=False)
            for (p0, px, py, pz) in int_regions.values():
                p0, px, py, pz = np.array(p0), np.array(px), np.array(py), np.array(pz)
                u = px - p0; v = py - p0; w = pz - p0
                c1 = np.dot(u, p0) - th <= np.dot(u, point) <= np.dot(u, px) + th
                c2 = np.dot(v, p0) - th <= np.dot(v, point) <= np.dot(v, py) + th
                c3 = np.dot(w, p0) - th <= np.dot(w, point) <= np.dot(w, pz) + th
                if c1 and c2 and c3:
                    return True
        except Exception:
            pass
    return False


def check_obj_upright(env, obj_name: str, th: float = 15.0) -> bool:
    """Check if object is upright (not tilted more than th degrees)."""
    from scipy.spatial.transform import Rotation as R
    actor = _get_obj_actor(env, obj_name)
    quat = actor.pose.q[0].cpu().numpy()  # wxyz
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # xyzw
    euler = r.as_euler("xyz", degrees=True)
    return bool(abs(euler[0]) < th and abs(euler[1]) < th)


# =====================================================================
# SAPIEN fixture joint state helpers
# (Replaces MuJoCo env.sim.model.joint_name2id / env.sim.data.qpos)
# =====================================================================

def _get_fixture_articulation(env, fixture):
    """
    Find the SAPIEN Articulation object for a RoboCasa fixture.
    Matches by fixture.name substring in articulation.name.
    """
    scene = env.scene
    arts = scene.get_all_articulations()
    fname = fixture.name
    for art in arts:
        if fname in art.name:
            return art
    return None


def _get_joint_qpos(env, fixture, joint_suffix):
    """
    Get the qpos of a joint on a fixture articulation.
    joint_suffix: e.g. 'doorhinge', 'slidejoint', 'leftdoorhinge'
    Returns float or None if not found.
    """
    art = _get_fixture_articulation(env, fixture)
    if art is None:
        return None
    joints = art.get_joints()
    qpos = art.get_qpos()
    target = joint_suffix  # look for suffix match
    for i, j in enumerate(joints):
        if j.name.endswith(joint_suffix) or joint_suffix in j.name:
            if i < qpos.shape[-1]:
                val = qpos[..., i]
                if hasattr(val, 'item'):
                    return float(val.flatten()[0])
                return float(val)
    return None


def _get_fixture_joint_qpos_by_suffix(art, joint_suffix):
    """Get qpos of articulation joint matching suffix. Returns float or None."""
    if art is None:
        return None
    joints = art.get_joints()
    qpos = art.get_qpos()
    for i, j in enumerate(joints):
        if joint_suffix in j.name:
            if i < qpos.shape[-1]:
                val = qpos[..., i]
                return float(val.flatten()[0]) if hasattr(val, 'flatten') else float(val)
    return None


def sapien_get_door_state(env, fixture):
    """
    SAPIEN replacement for SingleCabinet.get_door_state(env).
    Returns {'door': normalized_pct} or {'left': pct, 'right': pct} for double doors.
    """
    from mani_skill.utils.scene_builder.robocasa.utils.object_utils import normalize_joint_value
    cls = type(fixture).__name__
    art = _get_fixture_articulation(env, fixture)
    if art is None:
        return {'door': 0.0}

    if cls == 'DoubleCabinet':
        right_q = _get_fixture_joint_qpos_by_suffix(art, 'rightdoorhinge') or 0.0
        left_q = -(_get_fixture_joint_qpos_by_suffix(art, 'leftdoorhinge') or 0.0)
        return {
            'left': normalize_joint_value(left_q, 0, np.pi / 2),
            'right': normalize_joint_value(right_q, 0, np.pi / 2),
        }
    elif cls == 'Drawer':
        q = _get_fixture_joint_qpos_by_suffix(art, 'slidejoint') or 0.0
        return {'drawer': normalize_joint_value(q, 0, fixture.max_displacement if hasattr(fixture, 'max_displacement') else 0.4)}
    else:
        # SingleCabinet or generic
        q = _get_fixture_joint_qpos_by_suffix(art, 'doorhinge') or 0.0
        orientation = getattr(fixture, 'orientation', 'right')
        sign = -1 if orientation == 'left' else 1
        q = q * sign
        return {'door': normalize_joint_value(q, 0, np.pi / 2)}


def sapien_get_drawer_state(env, fixture):
    """SAPIEN replacement for Drawer.get_drawer_state(env)."""
    from mani_skill.utils.scene_builder.robocasa.utils.object_utils import normalize_joint_value
    art = _get_fixture_articulation(env, fixture)
    if art is None:
        return {'drawer': 0.0}
    q = _get_fixture_joint_qpos_by_suffix(art, 'slidejoint') or 0.0
    max_disp = getattr(fixture, 'max_displacement', 0.4)
    return {'drawer': normalize_joint_value(q, 0, max_disp)}


def sapien_get_knob_state(env, fixture, knob_location):
    """SAPIEN replacement for Stove.get_knobs_state() — single knob."""
    art = _get_fixture_articulation(env, fixture)
    q = _get_fixture_joint_qpos_by_suffix(art, f'knob_{knob_location}_joint') or 0.0
    q = q % (2 * np.pi)
    if q < 0:
        q += 2 * np.pi
    return q


def sapien_get_knobs_state(env, fixture):
    """SAPIEN replacement for Stove.get_knobs_state(env)."""
    from mani_skill.utils.scene_builder.robocasa.fixtures.stove import STOVE_LOCATIONS
    state = {}
    for loc in STOVE_LOCATIONS:
        knob_elem = fixture.knob_joints.get(loc)
        site_elem = fixture.burner_sites.get(loc)
        if knob_elem is None or site_elem is None:
            continue
        state[loc] = sapien_get_knob_state(env, fixture, loc)
    return state
