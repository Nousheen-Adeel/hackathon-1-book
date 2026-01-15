import os

# --- CONFIGURATION ---
# In folders ko ignore kiya jayega
IGNORE_DIRS = {
    'node_modules', '.git', '.next', '__pycache__', 'dist', 'build', 
    '.vscode', 'venv', 'env', '.idea', 'coverage', '.venv'
}

# In files ko ignore kiya jayega
IGNORE_FILES = {
    'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', '.DS_Store', 
    'thumbs.db', 'bun.lockb'
}

# Sirf in extensions wali files uthayi jayengi (Image/Video files hatane ke liye)
ALLOWED_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.scss', 
    '.json', '.md', '.txt', '.env.example', '.yml', '.yaml', '.sql', '.toml'
}

def collect_code(source_path, output_file):
    source_path = os.path.abspath(source_path)
    
    if not os.path.exists(source_path):
        print(f"❌ Error: Folder '{source_path}' nahi mila!")
        return

    with open(output_file, 'w', encoding='utf-8') as outfile:
        print(f"📂 Scanning: {source_path} ...\n")
        
        file_count = 0
        
        for root, dirs, files in os.walk(source_path):
            # Ignore Directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                if file in IGNORE_FILES:
                    continue
                
                # Check Extension
                _, ext = os.path.splitext(file)
                if ext.lower() not in ALLOWED_EXTENSIONS:
                    continue
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, source_path)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        
                        # Formatting for the text file
                        outfile.write(f"\n{'='*60}\n")
                        outfile.write(f"FILE: {rel_path}\n")
                        outfile.write(f"{'='*60}\n")
                        outfile.write(content + "\n")
                        
                        file_count += 1
                        print(f"✅ Added: {rel_path}")
                except Exception as e:
                    print(f"⚠️ Skipped {rel_path}: {e}")

    print(f"\n🎉 Success! Total {file_count} files saved to: {output_file}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # User se path maange ga
    print("--- PROJECT CODE COPIER ---")
    target_folder = input("Project ka Path paste karein: ").strip()
    
    # Output file ka naam (Script wale folder mein hi banegi)
    output_filename = "full_project_code.txt"
    
    # Agar user ne quotes (" ") laga diye hain path mein to hata do
    target_folder = target_folder.replace('"', '').replace("'", "")
    
    collect_code(target_folder, output_filename)
    
    input("\nPress Enter to exit...")