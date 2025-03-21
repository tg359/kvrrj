# A PowerShell script to install UV and create a new venv that matches the installed Ladybug Pollination environment.
# It then downloads the code from a Github repository and checks that the dependencies in the requirements.txt file match the dependencies in the pyproject.toml file.
# It then uses UV to create a venv for the repository.

# to remove UV for an ultra-clean install, follow the steps here - https://docs.astral.sh/uv/getting-started/installation/#uninstallation

# TODO - rename the repo to the resultant name
$REPO_NAME = "kvrrj"
$REPO_URL = "https://github.com/tg359/$REPO_NAME/archive/refs/heads/main.zip"
$TARGET_DIR = "$env:USERPROFILE\Documents\GitHub\target_dir"
$PROGRAMFILES = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::ProgramFiles)
$POLLINATION_UNINSTALLER_EXE = "$PROGRAMFILES\pollination\uninstall.exe"
$EXPECTED_POLLINATION_VERSION = "1.50.1.0"
$LBT_PYTHON_EXE = "$PROGRAMFILES\ladybug_tools\python\python.exe"
$EXPECTED_POLLINATION_PYTHON_VERSION = "3.10"

# go to user directory for consistent starting point
Set-Location $env:USERPROFILE

# failsafe to ensure the target directory is in a sensible location, and not Win32 or similar
if ($TARGET_DIR -ne "$env:USERPROFILE\Documents\GitHub\target_dir") {
    Write-Host "- Target directory is in a potentially dangerous location ($TARGET_DIR)." -ForegroundColor Red
    exit
}

# create target dir if it doesn't exist
if (!(Test-Path $TARGET_DIR)) {
    Write-Host "- Creating target directory $TARGET_DIR" -ForegroundColor Blue
    $null = New-Item -ItemType Directory -Path $TARGET_DIR  # assign to null to suppress output
}

# install UV gobally
Write-Host "- Installing UV" -ForegroundColor Blue
try {
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    Write-Host "- UV installed" -ForegroundColor Green
}
catch {
    Write-Host "- Failed to install UV." -ForegroundColor Red
    exit
}

# add UV to PATH
Write-Host "- Adding UV to PATH" -ForegroundColor Blue
try {
    $env:Path = "C:\Users\tgerrish\.local\bin;$env:Path"
    Write-Host "- UV added to PATH" -ForegroundColor Green
}
catch {
    Write-Host "- Failed to add UV to PATH." -ForegroundColor Red
    exit
}

# install global packages to UV
Write-Host "- Installing ruff" -ForegroundColor Blue
try {
    uv tool install ruff
    Write-Host "- ruff installed" -ForegroundColor Green
}
catch {
    Write-Host "- Failed to install ruff." -ForegroundColor Red
    exit
}
Write-Host "- Installing jupyterlab" -ForegroundColor Blue
try {
    uv tool install jupyterlab
    Write-Host "- jupyterlab installed" -ForegroundColor Green
}
catch {
    Write-Host "- Failed to install jupyterlab." -ForegroundColor Red
    exit
}

# check if Pollination is installed
Write-Host "- Checking for Pollination install" -ForegroundColor Blue
if (!(Test-Path $POLLINATION_UNINSTALLER_EXE)) {
    Write-Host "- Pollination is not installed." -ForegroundColor Red
    exit
} else {
    Write-Host "- Pollination is installed" -ForegroundColor Green
}

# check that the installed version of Pollination matches the expected version
Write-Host "- Checking Pollination version is $EXPECTED_POLLINATION_VERSION" -ForegroundColor Blue
$INSTALLED_POLLINATION_VERSION = (get-item $POLLINATION_UNINSTALLER_EXE).VersionInfo | ForEach-Object {("{0}.{1}.{2}.{3}" -f $_.ProductMajorPart,$_.ProductMinorPart,$_.ProductBuildPart,$_.ProductPrivatePart)}
if ($INSTALLED_POLLINATION_VERSION -ne $TARGET_POLLINATION_VERSION) {
    Write-Host "- Pollination version ($INSTALLED_POLLINATION_VERSION) is not $TARGET_POLLINATION_VERSION." -ForegroundColor Red
    exit
} else {
    Write-Host "- Pollination version is $EXPECTED_POLLINATION_VERSION" -ForegroundColor Green
}

# check that python is also installed with Pollination
Write-Host "- Checking for Python install with Pollination" -ForegroundColor Blue
if (!(Test-Path $LBT_PYTHON_EXE)) {
    Write-Host "- Python is not installed with Pollination." -ForegroundColor Red
    exit
} else {
    Write-Host "- Python is installed with Pollination" -ForegroundColor Green
}

# get the version of python associated with installed pollination
$LBT_PYTHON_VERSION = (get-item $LBT_PYTHON_EXE).VersionInfo | ForEach-Object {("{0}.{1}" -f $_.ProductMajorPart,$_.ProductMinorPart)}
Write-Host "- Checking Pollination Python version is $EXPECTED_POLLINATION_PYTHON_VERSION" -ForegroundColor Blue
if ($LBT_PYTHON_VERSION -ne $EXPECTED_POLLINATION_PYTHON_VERSION) {
    Write-Host "- Pollination Python version ($LBT_PYTHON_VERSION) is not $EXPECTED_POLLINATION_PYTHON_VERSION." -ForegroundColor Red
    exit
} else {
    Write-Host "- Pollination Python version is $EXPECTED_POLLINATION_PYTHON_VERSION" -ForegroundColor Green
}

# create a requirements.txt file from the pollination environment
$POLLINATION_PYTHON_REQUIREMENTS_TXT = "$TARGET_DIR\pollination_python_requirements.txt"
Write-Host "- Creating requirements.txt from Pollination Python environment" -ForegroundColor Blue
try {
    Start-Process -FilePath $LBT_PYTHON_EXE -NoNewWindow -ArgumentList "-m pip freeze" -RedirectStandardOutput $POLLINATION_PYTHON_REQUIREMENTS_TXT -Wait
    Write-Host "- requirements.txt created at $POLLINATION_PYTHON_REQUIREMENTS_TXT" -ForegroundColor Green
}
catch {
    Write-Host "- Failed to create requirements.txt from Pollination Python environment." -ForegroundColor Red
    exit
}

# download the repository containing the LBT_TK code into the target directory
Write-Host "- Downloading code from $REPO_URL to $TARGET_DIR" -ForegroundColor Blue
try {
    Invoke-RestMethod -Uri $REPO_URL -OutFile "$TARGET_DIR\$REPO_NAME.zip"
    Expand-Archive "$TARGET_DIR\$REPO_NAME.zip" $TARGET_DIR
    Rename-Item "$TARGET_DIR\$REPO_NAME-main" "$TARGET_DIR\$REPO_NAME"
    Remove-Item "$TARGET_DIR\$REPO_NAME.zip"
    Write-Host "- Code downloaded and extracted to $TARGET_DIR" -ForegroundColor Green
}
catch {
    Write-Host "- Failed to setup code from $REPO_URL in $TARGET_DIR" -ForegroundColor Red
    exit
}

# check that the version of python in requirements matches the version in the Pollination install
$REPO_PYTHON_VERSION_FILE = "$TARGET_DIR\$REPO_NAME\.python-version"
if (!(Test-Path $REPO_PYTHON_VERSION_FILE)) {
    Write-Host "- $REPO_PYTHON_VERSION_FILE does not exist." -ForegroundColor Red
    exit
}
$REPO_PYTHON_VERSION = Get-Content $REPO_PYTHON_VERSION_FILE
Write-Host "- Checking that the Python version in $REPO_PYTHON_VERSION_FILE matches the Pollination Python version" -ForegroundColor Blue
if ($REPO_PYTHON_VERSION -ne $EXPECTED_POLLINATION_PYTHON_VERSION) {
    Write-Host "- Python version in $REPO_PYTHON_VERSION_FILE ($REPO_PYTHON_VERSION) does not match the Pollination Python version ($EXPECTED_POLLINATION_PYTHON_VERSION)." -ForegroundColor Red
    exit
} else {
    Write-Host "- Python version in $REPO_PYTHON_VERSION_FILE matches the Pollination Python version" -ForegroundColor Green
}

# check that each of the packages in POLLINATION_PYTHON_REQUIREMENTS_TXT are listed in the pyproject.toml file as dependencies
$REPO_PYPROJECT_FILE = "$TARGET_DIR\kvrrj\pyproject.toml"
$POLLINATION_REQUIREMENTS = Get-Content $POLLINATION_PYTHON_REQUIREMENTS_TXT
$REPO_PYPROJECT = Get-Content $REPO_PYPROJECT_FILE
Write-Host "- Checking that the requirements in $POLLINATION_PYTHON_REQUIREMENTS_TXT match the requirements in $REPO_PYPROJECT_FILE" -ForegroundColor Blue
foreach ($line in $POLLINATION_REQUIREMENTS) {
    # if line doesnt start with either ladybug, honeybee, etc - skip
    if ($line -notmatch "^(ladybug|honeybee|lbt|dragonfly|queenbee|pollination)") {
        continue
    }
    $line = $line.Trim()
    if (-Not ($REPO_PYPROJECT -match $line)) {
        Write-Host "- $line is in $POLLINATION_PYTHON_REQUIREMENTS_TXT but not in $REPO_PYPROJECT_FILE" -ForegroundColor Red
        exit
    }
}

# use UV to create the venv for the repository
Write-Host "- Creating venv for $REPO_NAME" -ForegroundColor Blue
try {
    Set-Location $TARGET_DIR\$REPO_NAME
    uv sync
    Set-Location $env:USERPROFILE
}
catch {
    Write-Host "- Failed to create venv for $REPO_NAME" -ForegroundColor Red
    Set-Location $env:USERPROFILE
    exit
}

# add venv to global jupyter kernels in C:\Users\<USERNAME>\AppData\Roaming\jupyter\kernels
Write-Host "- Adding $REPO_NAME venv to global jupyter kernels ($env:USERPROFILE\AppData\Roaming\jupyter\kernels)" -ForegroundColor Blue
try {
    Set-Location $TARGET_DIR\$REPO_NAME
    uv run ipython kernel install --user --name=$REPO_NAME
    Set-Location $env:USERPROFILE
}
catch {
    Write-Host "- Failed to add $REPO_NAME venv to global jupyter kernels" -ForegroundColor Red
    Set-Location $env:USERPROFILE
    exit
}

# check for assets directory in the repository, and copy each file in assets into the installed kernel folder
$ASSETS_DIR = "$TARGET_DIR\$REPO_NAME\assets"
$KERNEL_DIR = "$env:USERPROFILE\AppData\Roaming\jupyter\kernels\$REPO_NAME"
if (!(Test-Path $ASSETS_DIR)) {
    Write-Host "- $ASSETS_DIR does not exist." -ForegroundColor Red
    exit
}
Write-Host "- Copying files from $ASSETS_DIR to $KERNEL_DIR" -ForegroundColor Blue
try {
    Copy-Item -Path $ASSETS_DIR\* -Destination $KERNEL_DIR -Recurse
    Write-Host "- Files copied to $KERNEL_DIR" -ForegroundColor Green
}
catch {
    Write-Host "- Failed to copy files from $ASSETS_DIR to $KERNEL_DIR" -ForegroundColor Red
    exit
}

# attempt to load code in VSCode
try {
    code "$TARGET_DIR\$REPO_NAME"
}
catch {
    Write-Host "- Failed to load code using VSCode" -ForegroundColor Red
    exit
}

