#!/usr/bin/env python3
"""Script to fix long lines in tools.py"""

import re

def fix_long_lines():
    with open('src/tools.py', 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for i, line in enumerate(lines):
        if len(line.strip()) > 100:
            # Check specific patterns and fix them
            if 'def ' in line and '(' in line and ')' in line:
                # Function definition
                line = re.sub(
                    r'def ([^(]+)\(([^)]+)\) -> ([^:]+):',
                    r'def \1(\n        \2\n    ) -> \3:',
                    line
                )
            elif '" if ' in line and ' else ' in line:
                # Complex ternary operations
                parts = line.split(' else ')
                if len(parts) == 2:
                    line = parts[0] + '\n        else ' + parts[1]
        
        fixed_lines.append(line)
    
    with open('src/tools.py', 'w') as f:
        f.writelines(fixed_lines)
    
    print("Fixed long lines in tools.py")

if __name__ == "__main__":
    fix_long_lines()