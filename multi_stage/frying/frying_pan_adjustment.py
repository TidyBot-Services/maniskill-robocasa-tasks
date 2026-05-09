from mani_skill.utils.registration import register_env
from robocasa_tasks import robocasa_utils as OU
from robocasa_tasks._base import *


@register_env("RoboCasa-Frying-Pan-Adjustment-v0", max_episode_steps=300, asset_download_ids=["RoboCasa"])
class FryingPanAdjustment(Kitchen):
    """
    Frying Pan Adjustment: composite task for Frying activity.

    Simulates the task of adjusting the frying pan on the stove.

    Steps:
        Move the pan from the current burner to another burner and turn on
        the burner.
    """

    def __init__(self, *args, **kwargs):
        self.start_loc = None
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=(0.3, 0.2))
        )
        self.init_robot_base_pos = self.stove

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """

        # First call super reset so that the pan is placed on the stove
        # then determine where it is placed and turn on the corresponding burner and update the start_loc
        super()._reset_internal()
        valid_knobs = self.stove.get_knobs_state(env=self).keys()
        pan_loc = OU.check_obj_location_on_stove(self, "obj", self.stove)

        for knob in valid_knobs:
            if pan_loc == knob:
                self.start_loc = pan_loc
                self.stove.set_knob_state(mode="on", knob=knob, env=self, rng=self.rng)
            else:
                self.stove.set_knob_state(mode="off", knob=knob, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.counter,
                    size=(0.60, 0.40),
                    pos=(0.0, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick and place the pan from the current burner to another burner and turn the burner on."
        return ep_meta

    def _check_success(self):
        # get the current location of the pan on the stove
        curr_loc = OU.check_obj_location_on_stove(self, "obj", self.stove)
        knobs_state = self.stove.get_knobs_state(env=self)
        knob_on_loc = False
        if curr_loc is not None and curr_loc in knobs_state:
            # check if burner is on where the pan is placed
            knob_on_loc = 0.35 <= np.abs(knobs_state[curr_loc]) <= 2 * np.pi - 0.35

        # return success if the pan is on a burner, the burner is on, and the burner is not the same as the start location
        return OU.gripper_obj_far(self) and knob_on_loc and curr_loc != self.start_loc
