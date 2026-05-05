param(
  [string]$Url = "https://github.com/Moobbot/LightOnOCR-2-1B/releases/download/v-1.0.0/model.zip",
  [string]$ZipPath = "model.zip",
  [string]$ModelFile = "model.safetensors",
  [int]$MaxAttempts = 100
)

$ErrorActionPreference = "Stop"

function Get-RemoteFileSize {
  param([string]$DownloadUrl)

  try {
    $headers = & curl.exe --silent --location --head $DownloadUrl
    if ($LASTEXITCODE -ne 0) {
      return $null
    }

    $lengths = @()
    foreach ($line in $headers) {
      if ($line -match '^[Cc]ontent-[Ll]ength:\s*(\d+)\s*$') {
        $lengths += [int64]$Matches[1]
      }
    }

    if ($lengths.Count -gt 0) {
      return $lengths[-1]
    }
  }
  catch {
    return $null
  }

  return $null
}

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

$expectedSize = Get-RemoteFileSize -DownloadUrl $Url
if ($expectedSize) {
  Write-Host "[INFO] Remote model.zip size: $expectedSize bytes."
}
else {
  Write-Host "[WARN] Khong lay duoc Content-Length tu remote. Se validate bang ZIP entry."
}

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
  $currentSize = 0
  if (Test-Path -LiteralPath $ZipPath) {
    $currentSize = (Get-Item -LiteralPath $ZipPath).Length
  }

  if ($expectedSize -and $currentSize -gt $expectedSize) {
    Write-Host "[WARN] model.zip lon hon remote size. Xoa file co the bi hong de tai lai."
    Remove-Item -LiteralPath $ZipPath -Force
    $currentSize = 0
  }

  Write-Host "[INFO] Tai model.zip attempt $attempt/$MaxAttempts (current bytes: $currentSize)"

  & curl.exe `
    --fail `
    --location `
    --continue-at - `
    --connect-timeout 30 `
    --speed-time 60 `
    --speed-limit 1024 `
    --output $ZipPath `
    $Url

  $curlExit = $LASTEXITCODE
  $newSize = 0
  if (Test-Path -LiteralPath $ZipPath) {
    $newSize = (Get-Item -LiteralPath $ZipPath).Length
  }

  if ($curlExit -ne 0) {
    Write-Host "[WARN] curl.exe exit code $curlExit. Bytes: $currentSize -> $newSize. Se resume lai."
  }
  else {
    Write-Host "[INFO] curl.exe finished. Bytes: $currentSize -> $newSize."
  }

  if (Test-ModelZip -Path $ZipPath -RequiredEntry $ModelFile) {
    $finalSize = (Get-Item -LiteralPath $ZipPath).Length
    Write-Host "[INFO] model.zip da tai du va ZIP hop le (bytes: $finalSize)."
    exit 0
  }

  if ($expectedSize -and $newSize -eq $expectedSize) {
    Write-Host "[WARN] model.zip du size nhung ZIP khong hop le. Xoa de tai lai tu dau."
    Remove-Item -LiteralPath $ZipPath -Force
  }

  Start-Sleep -Seconds 5
}

Write-Host "[ERROR] Khong the tai day du model.zip sau $MaxAttempts lan thu."
exit 1
