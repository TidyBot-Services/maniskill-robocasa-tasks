from mani_skill.utils.registration import register_env
from maniskill_tidyverse.robocasa_tasks import robocasa_utils as OU
from robocasa_tasks._base import *


@register_env("RoboCasa-Heat-Multiple-Water-v0", max_episode_steps=300, asset_download_ids=["RoboCasa"])
class HeatMultipleWater(Kitchen):
    """
    Heat Multiple Water: composite task for Boiling activity.

    Simulates the process of heating water in a pot and a kettle.

    Steps:
        Take the kettle from the cabinet and place it on a stove burner.
        Take the pot from the counter and place it on another stove burner.
        Turn both burners on.

    Args:
        init_robot_base_pos (str): Specifies a fixture to initialize the robot
            in front of. Default is "stove".
    """

    def __init__(self, init_robot_base_pos="stove", *args, **kwargs):
        super().__init__(init_robot_base_pos=init_robot_base_pos, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.ref_cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_TOP, ref=self.stove)
        )
        self.ref_counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )

        self.init_robot_base_pos = self.ref_cab

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("pot"),
                graspable=True,
                heatable=True,
                placement=dict(
                    fixture=self.ref_counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.50, 0.40),
                    pos=("ref", 0.0),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups=("kettle_non_electric"),
                graspable=True,
                placement=dict(
                    fixture=self.ref_cab,
                    size=(0.50, 0.30),
                    pos=(0, 0.0),
                ),
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick the kettle from the cabinet and place it on a stove burner. "
            "Then pick the pot from the counter and place it on another stove burner. "
            "Finally, turn both burners on."
        )
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()
        self.ref_cab.set_door_state(min=0.9, max=1.0, env=self, rng=self.rng)
        valid_knobs = self.stove.get_knobs_state(env=self).keys()

        for knob in valid_knobs:
            self.stove.set_knob_state(mode="off", knob=knob, env=self, rng=self.rng)

    def _check_success(self):

        pan_loc = OU.check_obj_location_on_stove(self, "obj", self.stove, threshold=0.15)
        kettle_loc = OU.check_obj_location_on_stove(self, "obj2", self.stove)

        # both objects placed on different parts of the stove
        successful_stove_placement = (
            (pan_loc is not None)
            and (kettle_loc is not None)
            and (pan_loc != kettle_loc)
        )

        return (
            successful_stove_placement
            and OU.gripper_obj_far(self)
            and OU.gripper_obj_far(self, "obj2")
        )
