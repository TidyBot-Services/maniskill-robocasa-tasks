from mani_skill.utils.registration import register_env
from maniskill_tidyverse.robocasa_tasks import robocasa_utils as OU
from robocasa_tasks._base import *
# fixtures imported via _base


@register_env("RoboCasa-Defrost-By-Category-v0", max_episode_steps=300, asset_download_ids=["RoboCasa"])
class DefrostByCategory(Kitchen):
    """
    Defrost By Category: composite task for Defrosting Food activity.

    Simulates the task of arranging and defrosting fruits and vegetables by type.

    Steps:
        Pick place all of the fruits in the running sink and all of the
        vegetables in a bowl on the counter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.5, 0.5))
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "There is a mixed pile of frozen fruits and vegetables on the counter. "
            "Locate all the frozen vegetables and place the items in a bowl on the counter. "
            "Take all the frozen fruits and defrost them in a running sink."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        # Place the four objects (two fruits, two vegetables)
        positions = [
            (0.0, -0.3),
            (0.0, 0.3),
            (-0.5, 0.0),
            (0.5, 0.0),
        ]
        self.rng.shuffle(positions)

        for i in range(4):
            cfgs.append(
                dict(
                    name="obj" + str(i),
                    obj_groups="fruit" if i <= 1 else "vegetable",
                    graspable=True,
                    placement=dict(
                        fixture=self.counter,
                        size=(0.40, 0.30),
                        pos=positions[i],
                        ensure_object_boundary_in_range=False,
                    ),
                )
            )

        # Bowl to place the vegetables in
        cfgs.append(
            dict(
                name="container",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    size=(0.50, 0.40),
                    pos=(0.0, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )
        return cfgs

    def _check_success(self):
        fruits_in_sink = OU.obj_inside_of(self, "obj0", self.sink) and OU.obj_inside_of(
            self, "obj1", self.sink
        )
        vegetables_in_bowl = OU.check_obj_in_receptacle(
            self, "obj2", "container"
        ) and OU.check_obj_in_receptacle(self, "obj3", "container")
        bowl_on_counter = OU.check_obj_fixture_contact(self, "container", self.counter)

        gripper_obj_far = True
        for i in range(4):
            gripper_obj_far = gripper_obj_far and OU.gripper_obj_far(
                self, obj_name="obj" + str(i)
            )

        return fruits_in_sink and vegetables_in_bowl and bowl_on_counter and gripper_obj_far
