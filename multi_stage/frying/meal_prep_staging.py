from mani_skill.utils.registration import register_env
from maniskill_tidyverse.robocasa_tasks import robocasa_utils as OU
from robocasa_tasks._base import *


@register_env("RoboCasa-Meal-Prep-Staging-v0", max_episode_steps=300, asset_download_ids=["RoboCasa"])
class MealPrepStaging(Kitchen):
    """
    Meal Prep Staging: composite task for Frying activity.

    Simulates the task of cooking various ingredients.

    Steps:
        Place the pans on different burners, then place the vegetable
        and meat on different pans.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=(0.3, 0.2))
        )
        self.init_robot_base_pos = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_name_1 = self.get_obj_lang("vegetable")
        obj_name_2 = self.get_obj_lang("meat")
        ep_meta["lang"] = (
            "Place both pans onto different burners. "
            f"Then place the {obj_name_1} and the {obj_name_2} on different pans."
        )
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _get_obj_cfgs(self):

        cfgs = []

        cfgs.append(
            dict(
                name="pan1",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.counter,
                    size=(0.60, 0.40),
                    pos=(0.0, -0.5),
                    rotation=0,
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="pan2",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.counter,
                    size=(0.60, 0.40),
                    pos=(0.0, 0.5),
                    rotation=0,
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups=("vegetable"),
                placement=dict(
                    fixture=self.counter,
                    size=(0.30, 0.30),
                    pos=(0.0, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups=("meat"),
                placement=dict(
                    fixture=self.counter,
                    size=(0.30, 0.30),
                    pos=(0.0, -0.2),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        return cfgs

    def _check_success(self):

        vegetable_on_pan1 = OU.check_obj_in_receptacle(self, "vegetable", "pan1")
        vegetable_on_pan2 = OU.check_obj_in_receptacle(self, "vegetable", "pan2")
        meat_on_pan1 = OU.check_obj_in_receptacle(self, "meat", "pan1")
        meat_on_pan2 = OU.check_obj_in_receptacle(self, "meat", "pan2")

        food_on_pans = (vegetable_on_pan1 and meat_on_pan2) or (
            vegetable_on_pan2 and meat_on_pan1
        )

        pan1_loc = OU.check_obj_location_on_stove(self, "pan1", self.stove)
        pan2_loc = OU.check_obj_location_on_stove(self, "pan2", self.stove)

        pans_on_stove = pan1_loc != None and pan2_loc != None
        pans_diff = pan1_loc != pan2_loc

        return pans_on_stove and pans_diff and food_on_pans
