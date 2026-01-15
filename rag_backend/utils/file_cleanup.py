import os
import hashlib
from pathlib import Path


def cleanup_project_files():
    """
    Cleans up duplicate markdown files and empty directories in the docs folder.
    - Traverses the '../book/docs' directory
    - Identifies and removes duplicate .md or .mdx files based on content hash
    - Deletes any empty folders left behind
    """
    docs_path = Path("../book/docs")
    
    if not docs_path.exists():
        print(f"Directory {docs_path} does not exist")
        return
    
    # Dictionary to store file hashes and their paths
    file_hashes = {}
    files_to_remove = []
    
    # Walk through all markdown files
    for file_path in docs_path.rglob("*"):
        if file_path.suffix.lower() in ['.md', '.mdx']:
            # Calculate MD5 hash of file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            if file_hash in file_hashes:
                # Duplicate found, mark for removal
                print(f"Duplicate found: {file_path} (duplicate of {file_hashes[file_hash]})")
                files_to_remove.append(file_path)
            else:
                # New unique file
                file_hashes[file_hash] = file_path
    
    # Remove duplicate files
    for file_path in files_to_remove:
        try:
            file_path.unlink()
            print(f"Removed duplicate file: {file_path}")
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")
    
    # Remove empty directories
    for dir_path in sorted(docs_path.rglob("*"), reverse=True):  # Reverse to remove child directories first
        if dir_path.is_dir() and not any(dir_path.iterdir()):
            try:
                dir_path.rmdir()
                print(f"Removed empty directory: {dir_path}")
            except Exception as e:
                print(f"Error removing directory {dir_path}: {e}")
    
    print("Cleanup completed.")


if __name__ == "__main__":
    cleanup_project_files()