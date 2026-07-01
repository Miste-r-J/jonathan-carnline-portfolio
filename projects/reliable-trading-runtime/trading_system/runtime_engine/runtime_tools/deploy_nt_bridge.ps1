param(
    [string]$SourcePath = (Join-Path $PSScriptRoot '..\integrations\ninjatrader\NinjaRepoBridge.cs'),
    [string]$TargetPath = (Join-Path $HOME 'Documents\NinjaTrader 8\bin\Custom\AddOns\NinjaRepoBridge.cs')
)

$ErrorActionPreference = 'Stop'

function Get-FileSha256([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return 'missing' }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$srcSha = Get-FileSha256 $SourcePath
if ($srcSha -eq 'missing') { throw "Source not found: $SourcePath" }

$dstShaBefore = Get-FileSha256 $TargetPath
Write-Host "NT_BRIDGE_DEPLOY|source=$SourcePath|source_sha256=$srcSha|target=$TargetPath|target_sha256_before=$dstShaBefore"

Copy-Item -LiteralPath $SourcePath -Destination $TargetPath -Force

$dstShaAfter = Get-FileSha256 $TargetPath
Write-Host "NT_BRIDGE_DEPLOY|target_sha256_after=$dstShaAfter"

if ($dstShaAfter -ne $srcSha) {
    throw "Deploy hash mismatch: source=$srcSha target=$dstShaAfter"
}

Write-Host "NT_BRIDGE_DEPLOY|status=ok"
