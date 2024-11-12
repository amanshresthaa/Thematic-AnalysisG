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
            if not callable(func):
                raise ValueError("The provided argument is not callable.")
            return func(*args, **kwargs)
        except ValueError as ve:
            logger.error(f"ValueError in {func.__name__}: {ve}", exc_info=True)
            return {"error": "Invalid input provided."}
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            return {"error": "An error occurred. Please try again later."}
    return wrapper
