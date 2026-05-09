from mani_skill.utils.registration import register_env
from robocasa_tasks import robocasa_utils as OU
from robocasa_tasks._base import *


@register_env("RoboCasa-Dessert-Upgrade-v0", max_episode_steps=300, asset_download_ids=["RoboCasa"])
class DessertUpgrade(Kitchen):
    """
    Dessert Upgrade: composite task for Serving Food activity.

    Simulates the task of serving dessert.

    Steps:
        Move the dessert items from the plate to the tray.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER_NON_CORNER, size=(1.0, 0.4))
        )
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = f"Move the dessert items from the plate to the tray."

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
                name="receptacle",
                obj_groups="tray",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    size=(0.60, 0.40),
                    pos=(0.0, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate1",
                obj_groups="plate",
                placement=dict(
                    fixture=self.counter,
                    size=(0.50, 0.40),
                    pos=(-0.5, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="dessert1",
                obj_groups="sweets",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.40, 0.30),
                    pos=(-0.3, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate2",
                obj_groups="plate",
                placement=dict(
                    fixture=self.counter,
                    size=(0.50, 0.40),
                    pos=(0.5, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="dessert2",
                obj_groups="sweets",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.40, 0.30),
                    pos=(0.3, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        return cfgs

    def _check_success(self):
        sweets_on_tray = OU.check_obj_in_receptacle(
            self, "dessert1", "receptacle"
        ) and OU.check_obj_in_receptacle(self, "dessert2", "receptacle")

        return sweets_on_tray and OU.gripper_obj_far(self, "receptacle")
