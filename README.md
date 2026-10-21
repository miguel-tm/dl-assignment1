# dl-assignment1

Deep Learning Assignment 1 (SCS 3546) — notebook-first regression experiments in TensorFlow/Keras.

## Project Structure

```text
.
├─ data/                           # place dataset files here
├─ notebooks/
│  └─ SCS_3546_Assignment_01.ipynb # single deliverable notebook
├─ outputs/                        # optional charts/tables exports
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Local Setup (Windows + VS Code)

### 1) Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell execution policy blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 2) Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Register/select kernel in VS Code

```powershell
python -m ipykernel install --user --name dl-assignment1 --display-name "Python (dl-assignment1)"
```

Then in VS Code:
- Open `notebooks/SCS_3546_Assignment_01.ipynb`
- Click **Select Kernel**
- Choose **Python (dl-assignment1)**

## Running in Google Colab

1. Open the notebook in Colab.
2. If needed, run:

```python
!pip install -q -r requirements.txt
```

3. Run all cells end-to-end.

> The notebook includes a first cell that auto-detects Colab vs local and sets paths without hardcoding `/content/` or Google Drive.

## Notes

- This repository is intentionally notebook-first.
- No `src/` package is used.
- Reusable logic should be implemented as helper functions inside the notebook.
