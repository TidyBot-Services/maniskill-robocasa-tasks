from mani_skill.utils.registration import register_env
from robocasa_tasks import robocasa_utils as OU
from robocasa_tasks._base import *


@register_env("RoboCasa-Simmering-Sauce-v0", max_episode_steps=300, asset_download_ids=["RoboCasa"])
class SimmeringSauce(Kitchen):
    """
    Simmering Sauce: composite task for Reheating Food activity.

    Simulates the task of simmering a sauce.

    Steps:
        Place the pan on a specific burner on the stove, then place the tomato and
        the onion in the pan and turn on the burner.

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
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=(0.5, 0.4))
        )
        self.init_robot_base_pos = self.stove

        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob"]
        else:
            valid_knobs = [
                k for (k, v) in self.stove.knob_joints.items() if v is not None
            ]
            if self.knob_id == "random":
                self.knob = self.rng.choice(list(valid_knobs))
            else:
                assert self.knob_id in valid_knobs
                self.knob = self.knob

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Place the pan on the {self.knob.replace('_', ' ')} burner on the stove. "
            f"Then place the tomato and the onion in the pan and turn on the {self.knob.replace('_', ' ')} burner."
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
                    fixture=self.counter,
                    # ensure_object_boundary_in_range=False because the pans handle is a part of the
                    # bounding box making it hard to place it if set to True
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(ref=self.stove, top_size=(0.50, 0.40)),
                    size=(0.25, 0.05),
                    pos=("ref", 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tomato",
                obj_groups="tomato",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.35, 0.2),
                    pos=("ref", -0.3),
                ),
            )
        )

        cfgs.append(
            dict(
                name="onion",
                obj_groups="onion",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.35, 0.2),
                    pos=("ref", 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        pan_on_stove = OU.check_obj_location_on_stove(self, "pan", self.stove) == self.knob
        tomato_in_pan = OU.check_obj_in_receptacle(self, "tomato", "pan")
        onion_in_pan = OU.check_obj_in_receptacle(self, "onion", "pan")

        knobs_state = self.stove.get_knobs_state(env=self)
        knob_value = knobs_state[self.knob]
        knob_on = 0.35 <= np.abs(knob_value) <= 2 * np.pi - 0.35

        return pan_on_stove and tomato_in_pan and onion_in_pan and knob_on
