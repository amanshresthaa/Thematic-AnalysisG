import os
import shutil
from pathlib import Path
import sys
from datetime import datetime

def setup_backup_directory() -> Path:
    """
    Create a new backup directory with timestamp in the same folder as the script.
    
    Returns:
        Path: Path to the created backup directory
    """
    script_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = script_dir / f"md_backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {backup_dir}")
    return backup_dir

def get_flattened_name(file_path: Path) -> str:
    """
    Generate a flattened filename based on the file's path.
    Example: docs/models/arch.md becomes models_arch.md
    
    Args:
        file_path (Path): Original file path
    
    Returns:
        str: New flattened filename
    """
    # Get the path relative to the docs directory
    rel_parts = file_path.parent.parts[file_path.parent.parts.index('docs')+1:]
    if rel_parts:
        # Join all directory names with underscore and append the filename
        return f"{'_'.join(rel_parts)}_{file_path.name}"
    return file_path.name

def copy_markdown_files(dest_dir: Path) -> tuple[int, list[str]]:
    """
    Copy all Markdown files from docs directory to destination directory with flattened names.
    
    Args:
        dest_dir (Path): Path to destination directory
    
    Returns:
        tuple[int, list[str]]: Number of files copied and list of any errors encountered
    """
    script_dir = Path(__file__).parent
    docs_path = script_dir / 'docs'
    
    if not docs_path.exists():
        return 0, [f"Source docs directory '{docs_path}' does not exist"]
    
    files_copied = 0
    errors = []
    
    try:
        # Track used names to handle potential duplicates
        used_names = set()
        
        for md_file in docs_path.rglob("*.md"):
            try:
                # Generate new flattened name
                new_name = get_flattened_name(md_file)
                
                # Handle potential name collisions
                base_name = new_name
                counter = 1
                while new_name in used_names:
                    name_parts = base_name.rsplit('.', 1)
                    new_name = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                    counter += 1
                
                used_names.add(new_name)
                
                # Copy the file with new name
                dest_file = dest_dir / new_name
                shutil.copy2(md_file, dest_file)
                
                # Print original path -> new name
                rel_path = md_file.relative_to(docs_path)
                print(f"Copied: {rel_path} -> {new_name}")
                files_copied += 1
                
            except Exception as e:
                errors.append(f"Error copying {md_file.name}: {str(e)}")
    
    except Exception as e:
        errors.append(f"Error accessing directory: {str(e)}")
    
    return files_copied, errors

def main():
    # Create destination directory with timestamp
    dest_dir = setup_backup_directory()
    
    # Copy files and handle results
    files_copied, errors = copy_markdown_files(dest_dir)
    
    # Print summary
    print(f"\nFiles copied: {files_copied}")
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"- {error}")
        
        if files_copied == 0:
            try:
                dest_dir.rmdir()
                print(f"\nRemoved empty directory: {dest_dir}")
            except:
                pass
        sys.exit(1)
    
    if files_copied == 0:
        print("\nNo markdown files found in docs directory.")
        try:
            dest_dir.rmdir()
            print(f"Removed empty directory: {dest_dir}")
        except:
            pass
        sys.exit(1)
    
    print(f"\nFiles successfully copied to: {dest_dir}")
    sys.exit(0)

if __name__ == "__main__":
    main()