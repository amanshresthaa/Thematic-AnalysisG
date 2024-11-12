import logging
import functools

logger = logging.getLogger(__name__)

def handle_exceptions(func):
    """
    Decorator to handle exceptions in functions.
    
    Args:
        func (callable): The function to wrap with exception handling.
    
    Returns:
        callable: The wrapped function with exception handling.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            return {"error": "An error occurred. Please try again later."}
    return wrapper
