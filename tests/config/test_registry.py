"""
Simple test registry to manage test cases.
"""
from typing import List, Optional
from .test_cases import BaseTestCase


class TestRegistry:
    """Simple registry for managing test cases."""
    
    def __init__(self):
        self.test_cases: List[BaseTestCase] = []
    
    def add_test_case(self, test_case: BaseTestCase):
        """Add a test case to the registry."""
        if not test_case.validate_params():
            raise ValueError(f"Invalid test case parameters: {test_case.test_id}")
        self.test_cases.append(test_case)
    
    def get_test_cases(self, test_group: Optional[str] = None) -> List[BaseTestCase]:
        """Get test cases, optionally filtered by group."""
        if test_group is None:
            return self.test_cases
        return [tc for tc in self.test_cases if tc.test_group == test_group]
    
    def get_test_case(self, test_id: str) -> Optional[BaseTestCase]:
        """Get specific test case by ID."""
        for tc in self.test_cases:
            if tc.test_id == test_id:
                return tc
        return None
    
    def get_summary(self) -> dict:
        """Get summary statistics of test cases."""
        groups = {}
        for tc in self.test_cases:
            groups[tc.test_group] = groups.get(tc.test_group, 0) + 1
        
        return {
            'total': len(self.test_cases),
            'by_group': groups
        }


# Global registry instance
TEST_REGISTRY = TestRegistry()
