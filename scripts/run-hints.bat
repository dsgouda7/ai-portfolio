@echo off
"c:\repos\ai-portfolio\.venv\Scripts\python.exe" "c:\repos\ai-portfolio\scripts\add-exercise-hints.py" > "c:\repos\ai-portfolio\hint-out.txt" 2>&1
echo Exit code: %errorlevel%
type "c:\repos\ai-portfolio\hint-out.txt"
