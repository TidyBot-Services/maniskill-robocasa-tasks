from mani_skill.utils.registration import register_env
from robocasa_tasks import robocasa_utils as OU
from robocasa_tasks._base import *


@register_env("RoboCasa-Kettle-Boiling-v0", max_episode_steps=300, asset_download_ids=["RoboCasa"])
class KettleBoiling(Kitchen):
    """
    Kettle Boiling: composite task for Brewing activity.

    Simulates the task of boiling water in a kettle.

    Steps:
        Take the kettle from the counter and place it on a stove burner.
        Turn the burner on.
    """

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_pos = self.stove
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=(0.2, 0.2))
        )

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("kettle_non_electric"),
                graspable=True,
                heatable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.35, 0.35),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="stove_distr",
                obj_groups=("pan", "pot"),
                placement=dict(
                    fixture=self.stove,
                    # ensure_object_boundary_in_range=False because the pans handle is a part of the
                    # bounding box making it hard to place it if set to True
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    # apply rotations so the handle doesnt stick too much
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the kettle from the counter and place it on a stove burner. Then turn the burner on."
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()
        valid_knobs = self.stove.get_knobs_state(env=self).keys()
        for knob in valid_knobs:
            self.stove.set_knob_state(mode="off", knob=knob, env=self, rng=self.rng)

    def _check_success(self):
        """
        Check if the kettle is placed on the stove burner and the burner is turned on.
        """
        kettle_loc = OU.check_obj_location_on_stove(self, "obj", self.stove, threshold=0.15)
        return kettle_loc is not None and OU.gripper_obj_far(self)
