#!/usr/bin/env python3
"""Script to fix variable names in tools.py"""

import re

def fix_tools_file():
    with open('src/tools.py', 'r') as f:
        content = f.read()
    
    # Replace specific patterns
    content = re.sub(r'\bvector_store\b', 'VECTOR_STORE', content)
    content = re.sub(r'\bprocessor\b', 'PROCESSOR', content)
    
    # But fix the class names and parameters that shouldn't be uppercase
    content = re.sub(r'ArticlePROCESSOR', 'ArticleProcessor', content)
    content = re.sub(r'VectorStore = None, ap: ArticlePROCESSOR', 'VectorStore = None, ap: ArticleProcessor', content)
    
    with open('src/tools.py', 'w') as f:
        f.write(content)
    
    print("Fixed tools.py variable names")

if __name__ == "__main__":
    fix_tools_file()