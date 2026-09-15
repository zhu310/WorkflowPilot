$ErrorActionPreference = "Stop"

$Root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$Frontend = Join-Path $Root "demo\frontend"

Push-Location $Frontend
try {
    if (-not (Test-Path -LiteralPath "node_modules")) {
        npm install
    }
    npm run build
}
finally {
    Pop-Location
}

python (Join-Path $Root "demo\server.py")
