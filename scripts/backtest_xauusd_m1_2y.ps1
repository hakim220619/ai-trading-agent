param(
    [double]$StartBalance = 50,
    [double]$FixedLot = 0.02,
    [int]$SessionStart = 13,
    [int]$SessionHours = 3,
    [int]$MaxSpreadPoints = 220,
    [int]$SlippagePoints = 20,
    [ValidateRange(1, 4)]
    [int]$MinSetups = 2
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Csv = Join-Path $ProjectRoot "data\XAUUSD_M1_2Y.csv"
$Report = Join-Path $ProjectRoot "logs\scalping_XAUUSD_M1_2Y_manual.json"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python virtual environment not found: $Python"
}

$PreviousLocation = Get-Location
try {
    Set-Location -LiteralPath $ProjectRoot

    Write-Host "Exporting two years of XAUUSD M1 candles from MT5..." -ForegroundColor Cyan
    & $Python -m app.mt5.export_data `
        --symbol XAUUSD `
        --timeframe M1 `
        --years 2 `
        --output $Csv
    if ($LASTEXITCODE -ne 0) {
        throw "MT5 candle export failed with exit code $LASTEXITCODE"
    }

    Write-Host "Running manual XAUUSD M1 backtest..." -ForegroundColor Cyan
    & $Python -m app.backtest.scalping_run `
        --strategy m1 `
        --csv $Csv `
        --symbol XAUUSD `
        --start-balance $StartBalance `
        --fixed-lot $FixedLot `
        --max-hold-bars 60 `
        --cooldown-bars 3 `
        --max-trades-per-day 3 `
        --max-consecutive-losses 2 `
        --max-spread-points $MaxSpreadPoints `
        --commission-per-lot-side 0 `
        --slippage-points $SlippagePoints `
        --weekdays-only `
        --session-start $SessionStart `
        --session-hours $SessionHours `
        --m1-min-setups $MinSetups `
        --output $Report
    if ($LASTEXITCODE -ne 0) {
        throw "Backtest failed with exit code $LASTEXITCODE"
    }

    Write-Host "Backtest complete. Report: $Report" -ForegroundColor Green
}
finally {
    Set-Location -LiteralPath $PreviousLocation
}
