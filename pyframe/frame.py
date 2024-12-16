from __future__ import annotations
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

class Frame:
    """
    Represents a orthonormal reference frame in 3D space with a defined position and orientation.

    By convention, the frame is defined from the world frame to the local frame. 
    Thats means, the coordinates of the rotation matrix are the coordinates of the local axis in the world coordinates.

    The Frame class is used to represent a reference frame in 3D space.
    The frame is defined by a position and an orientation. 
    The orientation can be defined using a quaternion, a rotation matrix, or Euler angles. 
    The frame can be direct or indirect, depending on the determinant of the rotation matrix.
    The frame can be converted to and from world coordinates using the from_world_to_local and from_local_to_world methods. 
    The frame can also be exported and imported as a dictionary using the dump and load methods.

    When the rotation matrix is set, the determinant is checked to determine if the frame is direct or indirect.

    However when the quaternion or the euleur angles is set, the frame is always considered direct. 
    Then the user must set the frame as indirect using the set_indirect method.
    In this case the rotation matrix is inverted to have a right-handed frame.

    In fact when the frame is indirect, the user can : 
    - set directly the indirect rotation matrix.
    - set the quaternion or the euler angles of the associated direct frame and then set the frame as indirect.

    The associated direct frame correspond to the frame with the same position but the local Z-axis inverted.

    Attributes:
    -----------
    position : np.ndarray, optional
        The position of the frame in 3D space with shape (3,1).
    quaternion : np.ndarray, optional
        The quaternion of the frame.
    rotation_matrix : np.ndarray, optional
        The rotation matrix of the frame.
    euler_angles : np.ndarray, optional 
        The Euler angles of the frame in radians.
    direct : bool, optional
        If the frame is direct or indirect.

    Properties:
    -----------
    position : np.ndarray
        Get or set the position of the frame in 3D space with shape (3,1).
    quaternion : np.ndarray
        Get or set the quaternion of the frame.
    rotation_matrix : np.ndarray
        Get the rotation matrix of the frame.
    euler_angles : np.ndarray
        Get the Euler angles of the frame in radians.
    homogeneous_matrix : np.ndarray
        Get the homogeneous transformation matrix of the frame.
    direct : bool
        Get or set if the frame is direct or indirect.
    x_axis : np.ndarray
        Get the X-axis of the frame with shape (3,1)
    y_axis : np.ndarray
        Get the Y-axis of the frame with shape (3,1)
    z_axis : np.ndarray
        Get the Z-axis of the frame with shape (3,1)
    blender_quaternion : np.ndarray
        Get the quaternion in Blender coordinate system.
    is_direct : bool
        Check if the frame is direct.
    is_indirect : bool
        Check if the frame is indirect.

    Examples:
    ---------
    >>> frame = Frame(position=np.array([1.0, 2.0, 3.0]), quaternion=np.array([0.5, 0.5, 0.5, 0.5]), direct=True)
    >>> frame.rotation_matrix
    """

    _tolerance = 1e-6

    # Initialization
    def __init__(
        self,
        *,
        position: Optional[np.ndarray] = None,
        quaternion: Optional[np.ndarray] = None,
        rotation_matrix: Optional[np.ndarray] = None,
        O3_project: bool = False,
        euler_angles: Optional[np.ndarray] = None,
        direct: bool = True,
        ) -> None:
        """
        Initialize a Frame object.

        Args:
            position (np.ndarray, optional): The position of the frame in 3D space with shape (3,1). Defaults to None.
            quaternion (np.ndarray, optional): The quaternion of the frame. Defaults to None.
            rotation_matrix (np.ndarray, optional): The rotation matrix of the frame. Defaults to None.

            euler_angles (np.ndarray, optional): The Euler angles of the frame in radians. Defaults to None.
            direct (bool, optional): If the frame is direct or indirect. Defaults to True.
        """
        # Set default values
        if sum([quaternion is not None, rotation_matrix is not None, euler_angles is not None]) > 1:
            raise ValueError("Only one of 'quaternion', 'rotation_matrix', or 'euler_angles' can be provided.")
        elif sum([quaternion is not None, rotation_matrix is not None, euler_angles is not None]) == 0:
            quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if position is None:
            position = np.zeros((3,1), dtype=np.float32)
        # Initialize the frame
        self.position = position
        self.direct = direct
        if quaternion is not None:
            self.quaternion = quaternion
        elif rotation_matrix is not None:
            if O3_project:
                self.set_O3_rotation_matrix(rotation_matrix)
            else:
                self.rotation_matrix = rotation_matrix
        elif euler_angles is not None:
            self.euler_angles = euler_angles

    # Properties getters and setters
    @property
    def position(self) -> np.ndarray:
        """Get or set the position of the frame in 3D space with shape (3,1)."""
        return self._position
    
    @position.setter
    def position(self, position: np.ndarray) -> None:
        self._position = np.array(position, dtype=np.float32).reshape((3,1))
    
    @property
    def quaternion(self) -> np.ndarray:
        """Get or set the quaternion of the frame."""
        quaternion = self._rotation.as_quat(scalar_first=True)
        quaternion = quaternion / np.linalg.norm(quaternion, ord=None)
        return quaternion
    
    @quaternion.setter
    def quaternion(self, quaternion: np.ndarray) -> None:
        quaternion = np.array(quaternion, dtype=np.float32).reshape((4,))
        norm = np.linalg.norm(quaternion, ord=None)
        if abs(norm) < self._tolerance:
            raise ValueError("Quaternion cannot have zero magnitude.")
        quaternion = quaternion / norm
        self._rotation = Rotation.from_quat(quaternion, scalar_first=True)
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get the rotation matrix of the frame."""
        rotation_matrix = self._rotation.as_matrix()
        if not self.direct:
            rotation_matrix[:, 2] = -rotation_matrix[:, 2]
        return rotation_matrix

    @rotation_matrix.setter
    def rotation_matrix(self, rotation_matrix: np.ndarray) -> None:
        rotation_matrix = np.array(rotation_matrix, dtype=np.float32).reshape((3,3))
        det = np.linalg.det(rotation_matrix)
        # Check if the determinant is close to 1
        if abs(abs(det) - 1.0) > self._tolerance:
            raise ValueError("Rotation matrix must have a determinant of 1 or -1.")
        # Check if the frame is direct or indirect
        if det < 0:
            rotation_matrix[:, 2] = -rotation_matrix[:, 2] # Invert the Z-axis to have a right-handed frame
            self.direct = False
        else:
            self.direct = True
        self._rotation = Rotation.from_matrix(rotation_matrix)

    @property
    def euler_angles(self) -> np.ndarray:
        """Get the Euler angles of the frame in radians."""
        return self._rotation.as_euler("XYZ", degrees=False)

    @euler_angles.setter
    def euler_angles(self, euler_angles: np.ndarray) -> None:
        euler_angles = np.array(euler_angles, dtype=np.float32).reshape((3,))
        self._rotation = Rotation.from_euler("XYZ", euler_angles, degrees=False)
    
    @property
    def homogeneous_matrix(self) -> np.ndarray:
        """Get the homogeneous transformation matrix of the frame."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation_matrix
        matrix[:3, 3] = self.position
        return matrix

    @homogeneous_matrix.setter
    def homogeneous_matrix(self, matrix: np.ndarray) -> None:
        homogeneous_matrix = np.array(matrix, dtype=np.float32).reshape((4,4))
        self.position = homogeneous_matrix[:3, 3]
        self.rotation_matrix = homogeneous_matrix[:3, :3]

    @property
    def direct(self) -> bool:
        """Get or set if the frame is direct or indirect."""
        if not isinstance(self._direct, bool):
            raise ValueError("Direct must be a boolean.")
        return self._direct
    
    @direct.setter
    def direct(self, direct: bool) -> None:
        if not isinstance(direct, bool):
            raise ValueError("Direct must be a boolean.")
        self._direct = direct

    @property
    def x_axis(self) -> np.ndarray:
        """Get the X-axis of the frame with shape (3,1)"""
        x_axis = self.rotation_matrix[:, 0].reshape((3,1))
        x_axis = x_axis / np.linalg.norm(x_axis, ord=None)
        return x_axis
    
    @property
    def y_axis(self) -> np.ndarray:
        """Get the Y-axis of the frame with shape (3,1)"""
        y_axis = self.rotation_matrix[:, 1].reshape((3,1))
        y_axis = y_axis / np.linalg.norm(y_axis, ord=None)
        return y_axis

    @property
    def z_axis(self) -> np.ndarray:
        """Get the Z-axis of the frame with shape (3,1)"""
        z_axis = self.rotation_matrix[:, 2].reshape((3,1))
        z_axis = z_axis / np.linalg.norm(z_axis, ord=None)
        return z_axis

    @property
    def blender_quaternion(self) -> np.ndarray:
        """Get the quaternion in Blender coordinate system."""
        frame = self.get_blender_frame()
        return frame.quaternion

    @property
    def is_direct(self) -> bool:
        """Check if the frame is direct."""
        return self.direct
    
    @property
    def is_indirect(self) -> bool:
        """Check if the frame is indirect."""
        return not self.direct

    # Private methods
    def _O3_projection(self, matrix: np.ndarray) -> np.ndarray:
        """
        Project a matrix to the orthogonal group O(3) using SVD and minimisation of the frobenius norm.

        Args:
            matrix (np.ndarray): A 3x3 matrix to be projected.

        Returns:
            np.ndarray: A 3x3 matrix in O(3).
        """
        matrix = np.array(matrix, dtype=np.float32).reshape((3,3))
        U, _, Vt = np.linalg.svd(matrix)
        orthogonal_matrix = np.dot(U, Vt)
        return orthogonal_matrix

    def _blender_conversion(self) -> (np.ndarray, np.ndarray):
        """
        Convert the frame to Blender coordinate system.

        Returns:
            tuple: A tuple containing the position and the rotation matrix in Blender coordinate system.
        """
        # Get the position
        blender_position = self.position
        # Convert into blender convention
        x_axis = self.x_axis
        y_axis = self.y_axis
        z_axis = self.z_axis
        blender_x_axis = y_axis
        blender_y_axis = x_axis
        blender_z_axis = -z_axis
        blender_rotation_matrix = np.column_stack((blender_x_axis, blender_y_axis, blender_z_axis))
        return blender_position, blender_rotation_matrix

    def _symmetric_conversion(self, point: np.ndarray, normal: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Convert the point and normal vector to the symmetric frame with respect to a plane.

        Args:
            point (np.ndarray): A point on the plane with shape (3,1).
            normal (np.ndarray): The normal vector of the plane with shape (3,1).

        Returns:
            tuple: A tuple containing the position and the rotation matrix of the symmetric frame.
        """
        # Compute the unit normal vector
        normal = np.array(normal, dtype=np.float32).reshape((3,1))
        norm = np.linalg.norm(normal, ord=None)
        if abs(norm) < self._tolerance:
            raise ValueError("Normal vector cannot have zero magnitude.")
        normal = normal / norm
        # Compute the reflected position
        symmetric_position = self.position - 2 * np.dot(normal, self.position - point) * normal
        # Compute the reflected rotation matrix
        x_axis = self.x_axis - 2 * np.dot(normal.T, self.x_axis) * normal
        y_axis = self.y_axis - 2 * np.dot(normal.T, self.y_axis) * normal
        z_axis = self.z_axis - 2 * np.dot(normal.T, self.z_axis) * normal
        symmetric_rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        return symmetric_position, symmetric_rotation_matrix

    def _inverse_conversion(self) -> (np.ndarray, np.ndarray):
        """
        By convention, the frame is defined from the world to the local.
        This method compute the frame defined from the local to the world frame.

        Returns:
            tuple: A tuple containing the position and the rotation matrix of the inverse frame.
        """
        # Compute the rotation matrix
        inverse_rotation_matrix = self.rotation_matrix.T
        # Compute the position
        inverse_position = self.from_world_to_local(point=np.array([0.0, 0.0, 0.0]))
        return inverse_position, inverse_rotation_matrix

    # Public methods
    def set_direct(self, direct: bool = True) -> None:
        """
        Set the frame as direct or indirect.

        Args:
            direct (bool, optional): Set the frame as direct or indirect. Defaults to True.
        """
        self.direct = direct
    
    def set_indirect(self, indirect: bool = True) -> None:
        """
        Set the frame as indirect or direct.

        Args:
            indirect (bool, optional): Set the frame as indirect or direct. Defaults to True.
        """
        self.direct = not indirect

    def set_O3_rotation_matrix(self, matrix: np.ndarray) -> None:
        """
        Set the rotation matrix of the frame in O(3).

        Args:
            matrix (np.ndarray): A 3x3 rotation matrix in O(3).
        """
        rotation_matrix = self._O3_projection(matrix)
        self.rotation_matrix = rotation_matrix

    def from_world_to_local(
        self, 
        *, 
        point: Optional[np.ndarray] = None, 
        vector: Optional[np.ndarray] = None
        ) -> Optional[np.ndarray]:
        """
        Convert a point or vector from world coordinates to local coordinates.

        if point is not None and vector is not None, the function will raise a ValueError.
        if point is None and vector is None, the function will return None.

        Args:
            point (np.ndarray, optional): Point in world coordinates with shape (3, N). Defaults to None.
            vector (np.ndarray, optional): Vector in world coordinates with shape (3, N). Defaults to None.

        Returns:
            np.ndarray: Point or vector in local coordinates with shape (3, N).

        Examples:
        >>> frame = Frame(position=np.array([1.0, 2.0, 3.0]), quaternion=np.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame.from_world_to_local(point=np.array([[1.0], [2.0], [3.0]]))
        """
        if point is not None and vector is not None:
            raise ValueError("Only one of 'point' or 'vector' can be provided.")
        if point is not None:
            point = np.array(point, dtype=np.float32).reshape((3, -1))
            local_point = np.dot(self.rotation_matrix.T, point - self.position)
            return local_point
        elif vector is not None:
            vector = np.array(vector, dtype=np.float32).reshape((3, -1))
            local_vector = np.dot(self.rotation_matrix.T, vector)
            return local_vector
        else:
            return None

    def from_local_to_world(
        self, 
        *, 
        point: Optional[np.ndarray] = None,
        vector: Optional[np.ndarray] = None
        ) -> Optional[np.ndarray]:
        """
        Convert a point or vector from local coordinates to world coordinates.

        if point is not None and vector is not None, the function will raise a ValueError.
        if point is None and vector is None, the function will return None.

        Args:
            point (np.ndarray, optional): Point in local coordinates with shape (3, N). Defaults to None.
            vector (np.ndarray, optional): Vector in local coordinates with shape (3, N). Defaults to None.

        Returns:
            np.ndarray: Point or vector in world coordinates with shape (3, N).

        Examples:
        >>> frame = Frame(position=np.array([1.0, 2.0, 3.0]), quaternion=np.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame.from_world_to_local(point=np.array([[1.0], [2.0], [3.0]]))
        """
        if point is not None and vector is not None:
            raise ValueError("Only one of 'point' or 'vector' can be provided.")
        if point is not None:
            point = np.array(point, dtype=np.float32).reshape((3, -1))
            world_point = self.position + np.dot(self.rotation_matrix, point)
            return world_point
        elif vector is not None:
            vector = np.array(vector, dtype=np.float32).reshape((3, -1))
            world_vector = np.dot(self.rotation_matrix, vector)
            return world_vector
        else:
            return None
    
    def get_blender_frame(self) -> Frame:
        """
        Get the frame in Blender coordinate system.

        Returns:
            Frame: The frame in Blender coordinate system.
        """
        blender_position, blender_rotation_matrix = self._blender_conversion()
        return Frame(position=blender_position, rotation_matrix=blender_rotation_matrix, O3_project=True)

    def apply_blender_frame(self) -> None:
        """
        Apply the Blender coordinate system to the frame.
        """
        _, blender_rotation_matrix = self._blender_conversion()
        self.set_O3_rotation_matrix(blender_rotation_matrix)

    def get_symmetric_frame(
        self, 
        point: np.ndarray, 
        normal: np.ndarray,
        ) -> Frame:
        """
        Get the symmetric frame of the current frame with respect to a plane defined by a point and a normal vector.

        Args:
            point (np.ndarray): A point on the plane with shape (3,1).
            normal (np.ndarray): The normal vector of the plane with shape (3,1).

        Returns:
            Frame: The symmetric frame of the current frame with respect to the plane.
        """
        symmetric_position, symmetric_rotation_matrix = self._symmetric_conversion(point, normal)
        return Frame(position=symmetric_position, rotation_matrix=symmetric_rotation_matrix, O3_project=True)
    
    def apply_symmetric_frame(
        self, 
        point: np.ndarray, 
        normal: np.ndarray
        ) -> None:
        """
        Apply a symmetry to the frame with respect to a plane defined by a point and a normal vector.

        Args:
            point (np.ndarray): A point on the plane with shape (3,1).
            normal (np.ndarray): The normal vector of the plane with shape (3,1).
        """
        symmetric_position, symmetric_rotation_matrix = self._symmetric_conversion(point, normal)
        self.set_O3_rotation_matrix(symmetric_rotation_matrix)
        self.position = symmetric_position

    def get_inverse_frame(self) -> Frame:
        """
        Get the inverse frame of the current frame.

        By convention, the frame is defined from the world to the local.
        This method compute the frame defined from the local to the world frame.

        Returns:
            Frame: The symmetric frame of the current frame with respect to the plane.
        """
        inverse_position, inverse_rotation_matrix = self._inverse_conversion()
        return Frame(position=inverse_position, rotation_matrix=inverse_rotation_matrix, O3_project=True)

    def apply_inverse_frame(self) -> None:
        """
        Apply a inversion to the frame.
        
        By convention, the frame is defined from the world to the local.
        This method compute the frame defined from the local to the world frame.
        """
        inverse_position, inverse_rotation_matrix = self._inverse_conversion()
        self.set_O3_rotation_matrix(inverse_rotation_matrix)
        self.position = inverse_position
    
    # Overridden methods
    def __repr__(self) -> str:
        """ String representation of the Frame object. """
        return f"Frame(position={self.position}, quaternion={self.quaternion}, direct={self.direct})"

    def __eq__(self, other: Frame) -> bool:
        """ Check if two Frame objects are equal. """
        return np.allclose(self.position, other.position) and np.allclose(self.quaternion, other.quaternion) and self.direct == other.direct

    # Load and dump methods
    def dump(self) -> dict:
        """
        Export the Frame's data as a dictionary.

        Returns:
            dict: A dictionary containing the Frame's position, quaternion, direct.
        """
        return {
            "position": self.position.tolist(),
            "quaternion": self.quaternion.tolist(),
            "direct": self.direct,
        }

    @classmethod
    def load(cls, data: dict) -> Frame:
        """
        Create a Frame instance from a dictionary.

        Args:
            data (dict): A dictionary containing the position, quaternion, direct.

        Returns:
            Frame: A new Frame instance initialized with the provided data.
        """
        # Check for required keys
        required_keys = {"position", "quaternion", "direct"}
        if not required_keys.issubset(data.keys()):
            raise ValueError(f"The dictionary must contain keys: {required_keys}")
        
        # Create the Frame instance
        return cls(
            position=data["position"],
            quaternion=data["quaternion"],
            direct=data["direct"],
        )













