"""
Provides the base class for all Connectors 
"""

from abc import ABC

class DBConnector(ABC):
    """
    The core class for all Connectors
    """

    connectorType: str = "Base"

    def __init__(self,
                project_id:str, 
                region:str):
        """
        Args:
            project_id (str | None): GCP Project Id.
            dataset_name (str): 
            TODO
        """
        self.project_id = project_id
        self.region = region 