import logging
import os
import asyncio
import aiofiles
from typing import List, Set, Tuple
import time
from pathlib import Path

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeFileHandler:
    def __init__(self, project_root: str, extensions: Set[str] = None):
        """
        Initialize the CodeFileHandler.
        
        Args:
            project_root (str): Root directory of the project
            extensions (Set[str]): Set of file extensions to include
        """
        self.project_root = project_root
        self.src_dir = os.path.join(project_root, '')
        self.extensions = extensions or {'.py', '.yaml', '.yml'}
        self.ignored_dirs = {'.venv', 'venv', 'env', '.env', 'myenv', '__pycache__', '.git', 'node_modules'}
        self.output_dir = os.path.join(project_root, 'output')

    async def get_code_files(self) -> List[Tuple[str, str]]:
        """
        Get all code files from the src directory.
        
        Returns:
            List[Tuple[str, str]]: List of (file_path, relative_path) tuples
        """
        logger.info(f"Scanning for code files in: {self.src_dir}")
        start_time = time.time()
        code_files = []
        
        try:
            for root, dirs, files in os.walk(self.src_dir):
                # Remove ignored directories
                dirs[:] = [d for d in dirs if d not in self.ignored_dirs]
                
                # Process files
                for file in files:
                    if file.endswith(tuple(self.extensions)):
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, self.src_dir)
                        code_files.append((file_path, relative_path))
                        logger.debug(f"Found code file: {relative_path}")
        
        except Exception as e:
            logger.error(f"Error scanning code files: {e}", exc_info=True)
            return []

        end_time = time.time()
        logger.info(f"Found {len(code_files)} code files in {end_time - start_time:.2f} seconds")
        return sorted(code_files, key=lambda x: x[1])  # Sort by relative path

    async def copy_code_to_file(self, code_files: List[Tuple[str, str]], output_file: str):
        """
        Copy code files to a single output file with proper organization.
        
        Args:
            code_files (List[Tuple[str, str]]): List of (file_path, relative_path) tuples
            output_file (str): Output file path
        """
        logger.info(f"Writing consolidated code to: {output_file}")
        start_time = time.time()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            async with aiofiles.open(output_file, 'w', encoding='utf-8') as outfile:
                # Write header
                await outfile.write("# Consolidated Source Code\n")
                await outfile.write("# " + "=" * 78 + "\n\n")

                current_module = None
                for file_path, relative_path in code_files:
                    # Get module name (first directory in relative path)
                    module = relative_path.split(os.sep)[0] if os.sep in relative_path else 'root'
                    
                    # Write module header if changed
                    if module != current_module:
                        await outfile.write(f"\n\n{'#' * 80}\n")
                        await outfile.write(f"# Module: {module}\n")
                        await outfile.write(f"{'#' * 80}\n\n")
                        current_module = module

                    # Write file header
                    await outfile.write(f"\n# File: {relative_path}\n")
                    await outfile.write("#" + "-" * 78 + "\n")

                    try:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                            content = await infile.read()
                            await outfile.write(content)
                            await outfile.write("\n")
                            logger.debug(f"Copied content from {relative_path}")
                    except Exception as e:
                        error_msg = f"# Error reading file {relative_path}: {str(e)}\n"
                        await outfile.write(error_msg)
                        logger.error(f"Error reading file {relative_path}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error writing to output file: {e}", exc_info=True)
            return

        end_time = time.time()
        logger.info(f"Code consolidation completed in {end_time - start_time:.2f} seconds")

async def main_async():
    """
    Main async function to handle code consolidation.
    """
    try:
        # Get project root (parent of src directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logger.info(f"Project root: {project_root}")

        # Initialize handler
        handler = CodeFileHandler(
            project_root=project_root,
            extensions={'.py', '.yaml', '.yml'}
        )

        # Create output directory if it doesn't exist
        output_dir = os.path.join(project_root, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Get code files
        code_files = await handler.get_code_files()
        
        if not code_files:
            logger.error("No code files found to process")
            return

        # Generate output file path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f'consolidated_code_{timestamp}.txt')

        # Copy code files
        await handler.copy_code_to_file(code_files, output_file)
        logger.info(f"Code consolidated successfully to: {output_file}")

    except Exception as e:
        logger.error(f"Error in main_async: {e}", exc_info=True)

def main():
    """
    Main entry point of the script.
    """
    try:
        asyncio.run(main_async())
        logger.info("Code consolidation completed successfully")
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    main()