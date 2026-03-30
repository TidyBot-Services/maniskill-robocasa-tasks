from mani_skill.utils.registration import register_env
from robocasa_tasks import robocasa_utils as OU
from robocasa_tasks._base import *


@register_env("RoboCasa-Setup-Frying-v0", max_episode_steps=300, asset_download_ids=["RoboCasa"])
class SetupFrying(Kitchen):
    """
    Setup Frying: composite task for Frying activity.

    Simulates the task of setting up the frying pan on the stove.

    Steps:
        Place the pan on the stove burner and turn the burner on.

    Args:
        cab_id (str): The id of the cabinet where the pan is placed.
    """

    def __init__(self, cab_id=FixtureType.DOOR_TOP_HINGE_DOUBLE, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.cab = self.register_fixture_ref(
            "cab", dict(id=self.cab_id, ref=self.stove)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )

        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Pick the pan from the cabinet and place it on the stove. "
            "Then turn on the stove burner for the pan."
        )
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()
        self.cab.set_door_state(min=0, max=0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pan",
                obj_groups="pan",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    # ensure_object_boundary_in_range=False because the pans handle is a part of the
                    # bounding box making it hard to place it if set to True
                    ensure_object_boundary_in_range=False,
                    size=(0.05, 0.02),
                    pos=(0, 0),
                    # apply a custom rotation for the pan so that it fits better in the cabinet
                    # (if the handle sticks out it may not fit)
                    rotation=(2 * np.pi / 8, 3 * np.pi / 8),
                ),
            )
        )

        # distractors
        for i in range(2):
            cfgs.append(
                dict(
                    name=f"distr_counter_{i}",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.cab,
                        ),
                        size=(0.50, 0.50),
                        pos=(0.0, -1.0),
                    ),
                )
            )
        cfgs.append(
            dict(
                name="distr_stove",
                obj_groups=("kettle_non_electric"),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the pan is placed on a stove burner and the burner is turned on.
        """
        pan_loc = OU.check_obj_location_on_stove(self, "pan", self.stove, threshold=0.15)
        return pan_loc is not None and OU.gripper_obj_far(self, "pan")
