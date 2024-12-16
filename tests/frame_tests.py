import unittest
import numpy as np

from pyframe import Frame, FrameBinder

class TestFrame(unittest.TestCase):

    def test_initialization(self):
        frame = Frame(position=np.array([1.0, 2.0, 3.0]), quaternion=np.array([0.5, 0.5, 0.5, 0.5]))
        self.assertTrue(np.allclose(frame.position, np.array([[1.0], [2.0], [3.0]])))
        self.assertTrue(np.allclose(frame.quaternion, np.array([0.5, 0.5, 0.5, 0.5]) / np.linalg.norm([0.5, 0.5, 0.5, 0.5])))

    def test_rotation_matrix(self):
        frame = Frame(rotation_matrix=np.eye(3))
        self.assertTrue(np.allclose(frame.rotation_matrix, np.eye(3)))

    def test_euler_angles(self):
        frame = Frame(euler_angles=np.array([0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(frame.euler_angles, np.array([0.0, 0.0, 0.0])))

    def test_from_world_to_local(self):
        frame = Frame(position=np.array([1.0, 2.0, 3.0]), quaternion=np.array([1.0, 0.0, 0.0, 0.0]))
        point_world = np.array([[2.0], [3.0], [4.0]])
        point_local = frame.from_world_to_local(point=point_world)
        self.assertTrue(np.allclose(point_local, np.array([[1.0], [1.0], [1.0]])))

    def test_from_local_to_world(self):
        frame = Frame(position=np.array([1.0, 2.0, 3.0]), quaternion=np.array([1.0, 0.0, 0.0, 0.0]))
        point_local = np.array([[1.0], [1.0], [1.0]])
        point_world = frame.from_local_to_world(point=point_local)
        self.assertTrue(np.allclose(point_world, np.array([[2.0], [3.0], [4.0]])))

    def test_dump_and_load(self):
        frame = Frame(position=np.array([1.0, 2.0, 3.0]), quaternion=np.array([0.5, 0.5, 0.5, 0.5]))
        data = frame.dump()
        loaded_frame = Frame.load(data)
        self.assertTrue(np.allclose(loaded_frame.position, frame.position))
        self.assertTrue(np.allclose(loaded_frame.quaternion, frame.quaternion))
        self.assertEqual(loaded_frame.direct, frame.direct)

class TestFrameBinder(unittest.TestCase):

    def setUp(self):
        self.binder = FrameBinder()
        self.frame1 = Frame(position=np.array([1.0, 2.0, 3.0]), quaternion=np.array([0.5, 0.5, 0.5, 0.5]))
        self.frame2 = Frame(position=np.array([4.0, 5.0, 6.0]), quaternion=np.array([0.5, 0.5, 0.5, 0.5]))

    def test_add_frame(self):
        self.binder.add_frame(self.frame1, "frame1")
        self.assertIn("frame1", self.binder.names)
        self.assertEqual(self.binder.get_frame("frame1"), self.frame1)

    def test_remove_frame(self):
        self.binder.add_frame(self.frame1, "frame1")
        self.binder.remove_frame("frame1")
        self.assertNotIn("frame1", self.binder.names)

    def test_recursive_remove_frame(self):
        self.binder.add_frame(self.frame1, "frame1")
        self.binder.add_frame(self.frame2, "frame2", link="frame1")
        self.binder.recursive_remove_frame("frame1")
        self.assertNotIn("frame1", self.binder.names)
        self.assertNotIn("frame2", self.binder.names)

    def test_set_name(self):
        self.binder.add_frame(self.frame1, "frame1")
        self.binder.set_name("frame1", "new_frame1")
        self.assertIn("new_frame1", self.binder.names)
        self.assertNotIn("frame1", self.binder.names)

    def test_set_link(self):
        self.binder.add_frame(self.frame1, "frame1")
        self.binder.add_frame(self.frame2, "frame2")
        self.binder.set_link("frame2", "frame1")
        self.assertEqual(self.binder.links["frame2"], "frame1")

    def test_get_worldcompose_frame(self):
        self.binder.add_frame(self.frame1, "frame1")
        world_frame = self.binder.get_worldcompose_frame("frame1")
        self.assertTrue(np.allclose(world_frame.position, self.frame1.position))
        self.assertTrue(np.allclose(world_frame.rotation_matrix, self.frame1.rotation_matrix))

    def test_get_compose_frame(self):
        self.binder.add_frame(self.frame1, "frame1")
        self.binder.add_frame(self.frame2, "frame2")
        composed_frame = self.binder.get_compose_frame("frame1", "frame2")
        self.assertTrue(np.allclose(composed_frame.position, self.frame2.position - self.frame1.position))

    def test_from_frame_to_frame(self):
        self.binder.add_frame(self.frame1, "frame1")
        self.binder.add_frame(self.frame2, "frame2")
        point = np.array([[1.0], [1.0], [1.0]])
        transformed_point = self.binder.from_frame_to_frame("frame1", "frame2", point=point)
        self.assertIsNotNone(transformed_point)

    def test_clear(self):
        self.binder.add_frame(self.frame1, "frame1")
        self.binder.clear()
        self.assertEqual(self.binder.num_frames, 0)

if __name__ == "__main__":
    unittest.main()