from mani_skill.utils.registration import register_env
from robocasa_tasks import robocasa_utils as OU
from robocasa_tasks._base import *


@register_env("RoboCasa-Pan-Transfer-v0", max_episode_steps=300, asset_download_ids=["RoboCasa"])
class PanTransfer(Kitchen):
    """
    Pan Transfer: composite task for Serving Food activity.

    Simulates the task of transferring vegetables from a pan to a plate.

    Steps:
        Pick up the pan and dump the vegetables in it onto the plate.
        Then, return the pan to the stove.
    """

    EXCLUDE_LAYOUTS = [0, 2, 4, 5]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_pos = self.stove
        self.dining_table = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=(0.3, 0.2))
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick up the pan and dump the vegetables in it onto the plate. "
            "Then return the pan to the stove."
        )
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pan",
                obj_groups="pan",
                placement=dict(
                    fixture=self.stove,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.60, 0.40),
                    pos=(0.0, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                graspable=False,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.60, 0.40),
                    pos=(0.0, -0.3),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )
        cfgs.append(
            dict(
                name="dstr_dining",
                obj_groups="all",
                exclude_obj_groups=["plate", "pan", "vegetable"],
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.40, 0.30),
                    pos=(0.0, 0.3),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )
        return cfgs

    def _check_success(self):
        vegetable_on_plate = OU.check_obj_in_receptacle(self, "vegetable", "plate")
        pan_on_stove = OU.check_obj_fixture_contact(
            self, "pan", self.stove
        )
        gripper_obj_far = OU.gripper_obj_far(
            self, "pan"
        ) and OU.gripper_obj_far(self, "vegetable")

        return vegetable_on_plate and pan_on_stove and gripper_obj_far
