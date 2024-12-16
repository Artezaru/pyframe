from typing import Optional, List, Dict

import numpy as np
from copy import copy as copycopy

from .frame import Frame

class FrameBinder:

    def __init__(self) -> None:
        self._frames = {}
        self._links = {}

    # Property getters and setters
    @property
    def names(self) -> List[str]:
        """Get a list of the names of the frames."""
        return list(self._frames.keys())
    
    @property
    def frames(self) -> Dict[str, Frame]:
        """Get a dictionary of the frames."""
        return self._frames

    @property
    def links(self) -> Dict[str, str]:
        """Get a dictionary of the links between frames."""
        return self._links
    
    @property
    def num_frames(self) -> int:
        """Get the number of frames in the FrameBinder."""
        return len(self._frames)
    
    # Private methods
    def _exist_name(self, name: str) -> bool:
        """Check if a frame with the given name exists."""
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        return name in self._frames

    def _world_frame(self) -> Frame:
        """Return the default world frame"""
        return Frame()

    # Public methods
    def add_frame(
        self, 
        frame: Frame,
        name: str,
        link: Optional[str] = None,
        copy: bool = False
        ) -> None:
        """
        Add a frame to the FrameBinder.

        Args:
            frame (Frame): The frame to be added.
            name (str): The name of the frame.
            link (str, optional): The name of the frame to link to. Defaults to None.
            copy (bool): Save a copy of the given frame. Defaults to False.

        Raises:
            TypeError: If the frame is not a Frame instance.
            ValueError: If the frame name already exists.
            ValueError: If the link name does not exist.
        """
        if not isinstance(frame, Frame):
            raise TypeError("frame is not Frame instance.")
        if not isinstance(copy, bool):
            raise TypeError("copy must be a boolean.")
        # Check if the frame name already exists
        if self._exist_name(name):
            raise ValueError(f"Frame with name '{name}' already exists.")
        if link is not None and link not in self._frames:
            raise ValueError(f"Link '{link}' does not exist.")
        # Add the frame to the FrameBinder
        if copy:
            frame = copycopy(frame)
        self._frames[name] = frame
        self._links[name] = link

    def remove_frame(self, name: str) -> None:
        """
        Remove a frame from the FrameBinder.

        The frames link to the removed frame will be set to None.

        Args:
            name (str): The name of the frame to be removed.

        Raises:
            ValueError: If the frame name does not exist.
        """
        if not self._exist_name(name):
            raise ValueError(f"Frame with name '{name}' does not exist.")
        for key, value in self._links.items():
            if value == name:
                self._links[key] = None
        del self._frames[name]
        del self._links[name]
        
    def recursive_remove_frame(self, name: str) -> None:
        """
        Recursively remove a frame and all its linked frames from the FrameBinder.

        Args:
            name (str): The name of the frame to be removed.

        Raises:
            ValueError: If the frame name does not exist.
        """
        if not self._exist_name(name):
            raise ValueError(f"Frame with name '{name}' does not exist.")
        to_remove = [name]
        while len(to_remove) > 0:
            name = to_remove.pop()
            # Adding the linked frames to the list of frames to remove
            for key, value in self._links.items():
                if value == name:
                    to_remove.append(key)
            # Removing the frame
            del self._frames[name]
            del self._links[name]
    
    def get_frame(self, name: str, copy: bool = False) -> Frame:
        """
        Get a frame from the FrameBinder.

        Args:
            name (str): The name of the frame to get.
            copy (bool): Get a copy of the given frame. Defaults to False.

        Returns:
            Frame: The frame with the given name.

        Raises:
            ValueError: If the frame name does not exist.
        """
        if not isinstance(copy, bool):
            raise TypeError("copy must be a boolean.")
        if not self._exist_name(name):
            raise ValueError(f"Frame with name '{name}' does not exist.")
        if copy:
            return copycopy(self._frames[name])
        return self._frames[name]
    
    def set_name(self, old_name: str, new_name: str) -> None:
        """
        Set a new name for a frame in the FrameBinder.

        Args:
            old_name (str): The old name of the frame.
            new_name (str): The new name of the frame.

        Raises:
            ValueError: If the frame name does not exist.
            ValueError: If the new name already exists.
        """
        # Check if the frame name already exists
        if not self._exist_name(old_name):
            raise ValueError(f"Frame with name '{old_name}' does not exist.")
        if self._exist_name(new_name):
            raise ValueError(f"Frame with name '{new_name}' already exists.")
        # Set the new name for the frame
        self._frames[new_name] = self._frames.pop(old_name)
        self._links[new_name] = self._links.pop(old_name)
        # Update the links
        for key, value in self._links.items():
            if value == old_name:
                self._links[key] = new_name
    
    def set_link(self, name: str, link: Optional[str] = None) -> None:
        """
        Set a link for a frame in the FrameBinder.

        Args:
            name (str): The name of the frame.
            link (str, optional): The name of the frame to link to. Defaults to None.

        Raises:
            ValueError: If the frame name does not exist.
            ValueError: If the link name does not exist.
        """
        if not self._exist_name(name):
            raise ValueError(f"Frame with name '{name}' does not exist.")
        if link is not None and link not in self._frames:
            raise ValueError(f"Link '{link}' does not exist.")
        self._links[name] = link

    def get_worldcompose_frame(self, name: Optional[str] = None) -> Frame:
        """
        The return frame is the transformation between the world frame and the frame with the given name.

        If the name is None, the function will return the world frame.

        Args:
            name (str, optional): The name of the frame to get. Defaults to None.

        Returns:
            Frame: The composed frame with the given name.
        
        Raises:
            ValueError: If the frame name does not exist.
        """
        if name is None:
            return self._world_frame()
        rotation_matrix = np.eye(3)
        position = np.zeros(3)
        while name is not None:
            frame = self.get_frame(name, copy=False)
            rotation_matrix = frame.from_local_to_world(vector=rotation_matrix)
            position = frame.from_local_to_world(point=position)
            name = self._links[name]
        return Frame(position=position, rotation_matrix=rotation_matrix, O3_project=True)

    def get_compose_frame(
        self, 
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        ) -> Frame:
        """
        The return frame is the transformation between the input frame and the output frame.

        if input_name is None, the function will return the worldcompose frame of the output frame.
        if output_name is None, the function will return the inverse worldcompose frame of the input frame.

        Args:
            input_name (str, optional): The name of the input frame. Defaults to None.
            output_name (str, optional): The name of the output frame. Defaults to None.

        Returns:
            Frame: The composed frame with the given name.

        Raises:
            ValueError: If the frame name does not exist.
        """
        if (input_name is None and output_name is None) or (input_name == output_name):
            return self._world_frame()
        input_frame = self.get_worldcompose_frame(input_name) # World -> Input
        input_frame.apply_inverse_frame() # Input -> World 
        output_frame = self.get_worldcompose_frame(output_name) # World -> Output
        rotation_matrix = input_frame.from_local_to_world(vector=output_frame.rotation_matrix)
        position = input_frame.from_local_to_world(point=output_frame.position)
        return Frame(position=position, rotation_matrix=rotation_matrix, O3_project=True)
    
    def from_frame_to_frame(
        self,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        *,
        point: Optional[np.ndarray] = None,
        vector: Optional[np.ndarray] = None,
        ) -> Optional[np.ndarray]:
        """
        Convert a point or vector from one frame to another frame.

        if point is not None and vector is not None, the function will raise a ValueError.
        if point is None and vector is None, the function will return None.

        if name is None, the function will return the point or vector in world coordinates.

        Args:
            input_name (str, optional): The name of the input frame. Defaults to None.
            output_name (str, optional): The name of the output frame. Defaults to None.
            point (np.ndarray, optional): Point in input frame coordinates with shape (3, N). Defaults to None.
            vector (np.ndarray, optional): Vector in input frame coordinates with shape (3, N). Defaults to None.

        Returns:
            np.ndarray: Point or vector in output frame coordinates with shape (3, N).
        """
        compose_frame = self.get_compose_frame(input_name=input_name, output_name=output_name)
        return compose_frame.from_world_to_local(point=point, vector=vector)

    def clear(self) -> None:
        """Clear all frames and links in the FrameBinder."""
        self._frames.clear()
        self._links.clear()
