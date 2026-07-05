from pathlib import Path
file_path=Path("sample.txt")
with open(file_path,"r",encoding="utf-8") as file:
    text=file.read()
print(text)
