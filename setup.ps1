# globals
$TARGET_DIR = Get-Location

# delete contents of the current directory, except for the setup.ps1 file
# ask for confirmation before deleting
Write-Host "This script will delete all files in the current directory, except for the setup.ps1 file." -ForegroundColor Yellow
$confirmation = Read-Host "Are you sure you want to continue? (y/n)"
if ($confirmation -ne "y") {
    Write-Host "- User cancelled script." -ForegroundColor Red
    exit
}
if (!(Test-Path "$TARGET_DIR\setup.ps1")) {
    Write-Host "- setup.ps1 file not found in the current directory, deleting probably a bad idea!" -ForegroundColor Red
    exit
}
Get-ChildItem -Path $TARGET_DIR | Where-Object { $_.Name -ne "setup.ps1" } | Remove-Item -Recurse -Force

# create BHoM directory if it doesn't exist, and set location to it
if (!(Test-Path $TARGET_DIR)) {
    New-Item -ItemType Directory -Path $TARGET_DIR
}
Set-Location $TARGET_DIR

# Install UV gobally
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Add UV to PATH
$env:Path = "C:\Users\tgerrish\.local\bin;$env:Path"

# Install Ruff
uv tool install ruff

# Install Jupyter
uv tool install jupyterlab

# # Install python venv for Python_Toolkit
# uv venv "python_toolkit" --python 3.12 --no-project

#######################################################################
#######################################################################

# ensure pollination is installed, and the correct version of pollination is installed
$PROGRAMFILES = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::ProgramFiles)
$LBT_PYTHON_EXE = "$PROGRAMFILES\ladybug_tools\python\python.exe"
$POLLINATION_UNINSTALLER_EXE = "$PROGRAMFILES\pollination\uninstall.exe"
if (!(Test-Path $POLLINATION_UNINSTALLER_EXE) -or !(Test-Path $LBT_PYTHON_EXE)) {
    Write-Host "- Pollination/ladybug is not installed." -ForegroundColor Red
    exit
}
$TARGET_POLLINATION_VERSION = "1.50.1.0"
$POLLINATION_VERSION = (get-item $POLLINATION_UNINSTALLER_EXE).VersionInfo | ForEach-Object {("{0}.{1}.{2}.{3}" -f $_.ProductMajorPart,$_.ProductMinorPart,$_.ProductBuildPart,$_.ProductPrivatePart)}
if ($POLLINATION_VERSION -ne $TARGET_POLLINATION_VERSION) {
    Write-Host "- Pollination version ($POLLINATION_VERSION) is not $TARGET_POLLINATION_VERSION." -ForegroundColor Red
    exit
}

# get the version of python associated with installed pollination
$LBT_PYTHON_EXE = "C:\Program Files\ladybug_tools\python\python.exe"
$LBT_PYTHON_VERSION = (get-item $LBT_PYTHON_EXE).VersionInfo | ForEach-Object {("{0}.{1}" -f $_.ProductMajorPart,$_.ProductMinorPart)}

# create the directory where code will be developed, and init
$PACKAGE_NAME = "kvrrj"
uv init --package --no-workspace --name $PACKAGE_NAME --python $LBT_PYTHON_VERSION --description "Toolkit for doing things ..." --author-from git

# # create python venv for LadybugTools_Toolkit
# uv venv "ladybugtools_toolkit" --python $LBT_PYTHON_VERSION --no-project

# create requirements.txt file from reference environment and install
$LBT_REQUIREMENTS_TXT = "$TARGET_DIR\ladybugtools_toolkit_requirements.txt"
Start-Process -FilePath $LBT_PYTHON_EXE -NoNewWindow -ArgumentList "-m pip freeze" -RedirectStandardOutput $LBT_REQUIREMENTS_TXT -Wait
uv add -r $LBT_REQUIREMENTS_TXT
Remove-Item $LBT_REQUIREMENTS_TXT

# add dev dependencies
uv add --dev pytest pytest-cov pylint ipykernel Sphinx sphinx_bootstrap_theme sphinxcontrib-fulltoc sphinxcontrib-napoleon

# add other dependencies
uv add case-converter dash dask dask[distributed] geopandas[all] meteostat openpyxl plotly pyarrow pyet scikit-learn scipy tables tqdm xlrd

# create a vscode directory
$VSCODE_DIR = "$TARGET_DIR\.vscode"
if (!(Test-Path $VSCODE_DIR)) {
    New-Item -ItemType Directory -Path $VSCODE_DIR
}

# create a settings json file inside the vscode_dir, from a referenced file on github
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ArjanCodes/examples/refs/heads/main/2024/vscode_python/.vscode/settings.json" -OutFile "$VSCODE_DIR\settings.json"

# create an extenmsions json file inside the vscode_dir, from a referenced file on github
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ArjanCodes/examples/refs/heads/main/2024/vscode_python/.vscode/extensions.json" -OutFile "$VSCODE_DIR\extensions.json"

# create a test directory
$TEST_DIR = "$TARGET_DIR\tests"
if (!(Test-Path $TEST_DIR)) {
    New-Item -ItemType Directory -Path $TEST_DIR
}
# add a test file
New-Item -ItemType File -Path "$TEST_DIR\test_main.py"
# add following text to the test file
Set-Content -Path "$TEST_DIR\test_main.py" -Value "from $PACKAGE_NAME import main`r`n`r`n`r`ndef test_main():`r`n    main()`r`n"

# append the following to the pyproject.toml file
$PYPROJECT_TOML = "$TARGET_DIR\pyproject.toml"
$PYPROJECT_TOML_CONTENT = Get-Content $PYPROJECT_TOML
$PYPROJECT_TOML_CONTENT += "`r`n[tool.pytest.ini_options]`r`npythonpath = `"src`""
Set-Content -Path $PYPROJECT_TOML -Value $PYPROJECT_TOML_CONTENT

# add venv to jupyter
uv run ipython kernel install --user --name=$PACKAGE_NAME --env VIRTUAL_ENV "$TARGET_DIR\.venv"

# run jupyter
# uv tool run jupyter lab
