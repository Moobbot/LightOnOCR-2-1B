param(
  [string]$Url = "https://github.com/Moobbot/LightOnOCR-2-1B/releases/download/v-1.0.0/model.zip",
  [string]$ZipPath = "model.zip",
  [string]$ModelFile = "model.safetensors",
  [int]$MaxAttempts = 100
)

$ErrorActionPreference = "Stop"

function Test-ModelZip {
  param([string]$Path, [string]$RequiredEntry)

  if (-not (Test-Path -LiteralPath $Path)) {
    return $false
  }

  try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
    $zip = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath $Path))
    try {
      foreach ($entry in $zip.Entries) {
        if ($entry.FullName -eq $RequiredEntry -and $entry.Length -gt 0) {
          return $true
        }
      }
      return $false
    }
    finally {
      $zip.Dispose()
    }
  }
  catch {
    return $false
  }
}

if (Test-Path -LiteralPath $ModelFile) {
  Write-Host "[INFO] Da co file $ModelFile. Bo qua buoc tai model."
  exit 0
}

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
  Write-Host "[ERROR] Khong tim thay curl.exe de resume download."
  exit 1
}

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
  $currentSize = 0
  if (Test-Path -LiteralPath $ZipPath) {
    $currentSize = (Get-Item -LiteralPath $ZipPath).Length
  }

  Write-Host "[INFO] Tai model.zip attempt $attempt/$MaxAttempts (current bytes: $currentSize)"

  & curl.exe `
    --location `
    --continue-at - `
    --retry 10 `
    --retry-delay 5 `
    --retry-all-errors `
    --output $ZipPath `
    $Url

  $curlExit = $LASTEXITCODE
  if ($curlExit -ne 0) {
    Write-Host "[WARN] curl.exe exit code $curlExit. Se resume lai neu chua vuot qua gioi han."
  }

  if (Test-ModelZip -Path $ZipPath -RequiredEntry $ModelFile) {
    $finalSize = (Get-Item -LiteralPath $ZipPath).Length
    Write-Host "[INFO] model.zip da tai du va ZIP hop le (bytes: $finalSize)."
    exit 0
  }

  Start-Sleep -Seconds 5
}

Write-Host "[ERROR] Khong the tai day du model.zip sau $MaxAttempts lan thu."
exit 1
