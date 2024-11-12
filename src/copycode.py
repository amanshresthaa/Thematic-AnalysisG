import logging
import os
import asyncio
import aiofiles
from typing import List
from utils.logger import setup_logging
import time

logger = logging.getLogger(__name__)

class CodeFileHandler:
    def __init__(self, root_dir: str, extensions: List[str] = None):
        self.root_dir = root_dir
        self.extensions = extensions

    async def get_code_files(self) -> List[str]:
        logger.debug(f"Entering get_code_files with root_dir='{self.root_dir}' and extensions={self.extensions}.")
        start_time = time.time()
        code_files = []
        try:
            logger.debug(f"Walking through directory: {self.root_dir}")
            for root, dirs, files in os.walk(self.root_dir):
                ignored_dirs = {'.venv', 'venv', 'env', '.env', 'myenv'}
                dirs[:] = [d for d in dirs if d not in ignored_dirs]
                for ignored in ignored_dirs:
                    if ignored in dirs:
                        dirs.remove(ignored)
                        logger.debug(f"Skipping directory: {os.path.join(root, ignored)}")

                for file in files:
                    if self.extensions is None or file.endswith(tuple(self.extensions)):
                        file_path = os.path.join(root, file)
                        code_files.append(file_path)
                        logger.debug(f"Found code file: {file_path}")
        except Exception as e:
            logger.error(f"Error while retrieving code files: {e}", exc_info=True)

        logger.info(f"Total code files found: {len(code_files)}")
        end_time = time.time()
        logger.debug(f"Exiting get_code_files method. Time taken: {end_time - start_time:.2f} seconds.")
        return code_files

    async def copy_code_to_file(self, code_files: List[str], output_file: str):
        logger.debug(f"Entering copy_code_to_file with output_file='{output_file}'.")
        start_time = time.time()
        try:
            logger.info(f"Writing consolidated code to '{output_file}'")
            async with aiofiles.open(output_file, 'w', encoding='utf-8') as outfile:
                for filepath in code_files:
                    await outfile.write(f"\n\n# File: {filepath}\n")
                    try:
                        async with aiofiles.open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
                            content = await infile.read()
                            await outfile.write(content)
                        logger.debug(f"Copied content from '{filepath}'")
                    except Exception as e:
                        await outfile.write(f"# Error reading file {filepath}: {e}\n")
                        logger.error(f"Error reading file '{filepath}': {e}", exc_info=True)
                    await outfile.write("\n" + "#" * 80 + "\n")
            logger.info(f"All code has been copied to '{output_file}'")
        except Exception as e:
            logger.error(f"Error during copying code to file '{output_file}': {e}", exc_info=True)
        end_time = time.time()
        logger.debug(f"Exiting copy_code_to_file method. Time taken: {end_time - start_time:.2f} seconds.")

async def main_async():
    logger.debug("Entering main_async function.")
    start_time = time.time()
    try:
        script_directory = os.path.dirname(os.path.abspath(__file__))
        logger.debug(f"Script directory: {script_directory}")

        source_directory = script_directory

        output_file = os.path.join(script_directory, 'consolidated_code.txt')

        file_extensions = ['.py', '.yaml']
        logger.debug(f"File extensions to include: {file_extensions}")

        code_handler = CodeFileHandler(source_directory, file_extensions)
        code_files = await code_handler.get_code_files()

        await code_handler.copy_code_to_file(code_files, output_file)
    except Exception as e:
        logger.error(f"Unexpected error in main_async: {e}", exc_info=True)
    end_time = time.time()
    logger.debug(f"Exiting main_async function. Total time taken: {end_time - start_time:.2f} seconds.")

def main():
    logger.debug("Entering main function.")
    start_time = time.time()
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.warning("Code copying interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error during code copying: {e}", exc_info=True)
    end_time = time.time()
    logger.debug(f"Exiting main function. Total time taken: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
