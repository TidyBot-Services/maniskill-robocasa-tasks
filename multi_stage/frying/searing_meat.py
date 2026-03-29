from mani_skill.utils.registration import register_env
from maniskill_tidyverse.robocasa_tasks import robocasa_utils as OU
from robocasa_tasks._base import *


@register_env("RoboCasa-Searing-Meat-v0", max_episode_steps=300, asset_download_ids=["RoboCasa"])
class SearingMeat(Kitchen):
    """
    Searing Meat: composite task for Frying activity.

    Simulates the task of searing meat.

    Steps:
        Place the pan on the specified burner on the stove,
        then place the meat on the pan and turn the burner on.

    Args:
        knob_id (str): The id of the knob who's burner the pan will be placed on.
            If "random", a random knob is chosen.
    """

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )

        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_TOP, ref=self.stove)
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_name = self.get_obj_lang("meat")
        ep_meta["lang"] = (
            f"Grab the pan from the cabinet and place it on the {self.knob.replace('_', ' ')} burner on the stove. "
            f"Then place the {meat_name} on the stove and turn the burner on."
        )
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()

        valid_knobs = self.stove.get_knobs_state(env=self).keys()
        if self.knob_id == "random":
            self.knob = self.rng.choice(list(valid_knobs))
        else:
            assert self.knob_id in valid_knobs
            self.knob = self.knob

        self.stove.set_knob_state(mode="off", knob=self.knob, env=self, rng=self.rng)
        self.cab.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pan",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.cab,
                    ensure_object_boundary_in_range=False,
                    pos=(0.0, -0.3),
                    size=(0.4, 0.02),
                    rotation=np.pi / 2,
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                heatable=True,
                placement=dict(
                    fixture=self.counter,
                    loc="nn",
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="meat")
        pan_loc = OU.check_obj_location_on_stove(self, "pan", self.stove, threshold=0.15) == self.knob
        meat_in_pan = OU.check_obj_in_receptacle(self, "meat", "pan", th=0.07)
        return gripper_obj_far and pan_loc and meat_in_pan
