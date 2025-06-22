"""
Function registry for managing available functions.
"""
import logging
from typing import Dict, Any, Callable, List

from llm_api.core.functions.builtin.time import TIME_FUNCTION
from llm_api.core.functions.builtin.weather import WEATHER_FUNCTION

logger = logging.getLogger(__name__)


class FunctionRegistry:
    """Registry for managing available functions."""
    
    def __init__(self):
        """Initialize the function registry."""
        self._functions: Dict[str, Dict[str, Any]] = {}
        self._register_builtin_functions()
    
    def _register_builtin_functions(self) -> None:
        """Register built-in functions."""
        self.register_function(TIME_FUNCTION)
        self.register_function(WEATHER_FUNCTION)
        logger.info(f"Registered {len(self._functions)} built-in functions")
    
    def register_function(self, function_spec: Dict[str, Any]) -> None:
        """
        Register a function with the registry.
        
        Args:
            function_spec: Dictionary containing function metadata and callable
        """
        name = function_spec["name"]
        self._functions[name] = function_spec
        logger.debug(f"Registered function: {name}")
    
    def get_function(self, name: str) -> Dict[str, Any]:
        """
        Get a function by name.
        
        Args:
            name: Function name
            
        Returns:
            Function specification dictionary
            
        Raises:
            KeyError: If function not found
        """
        if name not in self._functions:
            raise KeyError(f"Function '{name}' not found")
        return self._functions[name]
    
    def list_functions(self) -> List[str]:
        """List all registered function names."""
        return list(self._functions.keys())
    
    def get_function_specs(self) -> List[Dict[str, Any]]:
        """
        Get function specifications for OpenAI-style function calling.
        
        Returns:
            List of function specifications (without the callable)
        """
        specs = []
        for func_spec in self._functions.values():
            spec = {
                "name": func_spec["name"],
                "description": func_spec["description"],
                "parameters": func_spec["parameters"]
            }
            specs.append(spec)
        return specs
    
    def call_function(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        Call a function with the given arguments.
        
        Args:
            name: Function name
            arguments: Function arguments
            
        Returns:
            Function result as string
            
        Raises:
            KeyError: If function not found
            Exception: If function call fails
        """
        func_spec = self.get_function(name)
        func_callable = func_spec["function"]
        
        try:
            result = func_callable(**arguments)
            logger.debug(f"Function '{name}' called successfully")
            return str(result)
        except Exception as e:
            logger.error(f"Error calling function '{name}': {e}")
            raise


# Global function registry instance
function_registry = FunctionRegistry() 