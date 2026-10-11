# Thin wrapper: powershell -ExecutionPolicy Bypass -File tools\arctic-batch\arctic-batch.ps1 run <images...> -o OUT --cli ... --replay ...
# Not run on Windows yet. Requires Python 3 on PATH (py -3 or python).
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command py -ErrorAction Stop }
& $py.Source (Join-Path $PSScriptRoot 'arctic_batch.py') @args
exit $LASTEXITCODE
