#!/usr/bin/env python3
"""Script to fix trailing whitespace and missing final newlines in Python files."""

import os
import re

def fix_file_formatting(file_path):
    """Fix trailing whitespace and ensure final newline in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove trailing whitespace from each line
        lines = content.splitlines()
        cleaned_lines = [line.rstrip() for line in lines]
        
        # Join lines and ensure final newline
        cleaned_content = '\n'.join(cleaned_lines)
        if cleaned_content and not cleaned_content.endswith('\n'):
            cleaned_content += '\n'
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print(f"Fixed: {file_path}")
        return True
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Main function to fix all Python files."""
    dirs_to_process = ['src', 'api', 'scripts', 'tests']
    files_fixed = 0
    
    for directory in dirs_to_process:
        if os.path.exists(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        if fix_file_formatting(file_path):
                            files_fixed += 1
    
    print(f"\nTotal files fixed: {files_fixed}")

if __name__ == "__main__":
    main()