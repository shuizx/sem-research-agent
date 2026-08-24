[CmdletBinding()]
param(
    [ValidateSet("agent", "pipeline", "research")]
    [string]$Mode = "agent",

    [string]$WorkspaceRoot = "",

    [switch]$SampleMode,

    [switch]$AutoApproveSample,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$vroRepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $vroRepositoryRoot

if ([string]::IsNullOrWhiteSpace($env:UV_CACHE_DIR)) {
    $env:UV_CACHE_DIR = Join-Path $env:TEMP "sem-agent-uv-cache"
}
if ([string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
    $WorkspaceRoot = Join-Path $vroRepositoryRoot "var\live-runs"
}
if ($Mode -ne "pipeline" -and $AutoApproveSample) {
    throw "-AutoApproveSample is available only for the local pipeline sample."
}
if ($Mode -ne "agent" -and $SampleMode) {
    throw "-SampleMode is available only for the conversational interface."
}

$vroTimestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$vroWorkflowId = "sem-agent-$Mode-$vroTimestamp"
$vroRunRoot = Join-Path $WorkspaceRoot $vroWorkflowId
$vroEnvFile = Join-Path $vroRepositoryRoot ".env"

if (-not $SampleMode -and -not $DryRun -and
    [string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY) -and
    -not (Test-Path -LiteralPath $vroEnvFile)) {
    throw "DashScope configuration missing. Copy .env.example to .env and set DASHSCOPE_API_KEY."
}

if ($Mode -eq "agent") {
    $vroAgentMode = if ($SampleMode) { "fixture" } else { "live" }
    $vroRouterMode = if ($SampleMode) { "fixture" } else { "live" }
    $vroArguments = @(
        "run", "python", "-m", "vision_research_ops.cli.agent",
        "--mode", $vroAgentMode,
        "--router", $vroRouterMode,
        "--workspace", $vroRunRoot
    )
}
elseif ($Mode -eq "pipeline") {
    $vroArguments = @(
        "run", "python", "-m", "vision_research_ops.cli.pipeline", "sample",
        "--mode", "fixture",
        "--adaptation-planner", "dashscope",
        "--workspace", $vroRunRoot,
        "--workflow-id", $vroWorkflowId
    )
    if ($AutoApproveSample) {
        $vroArguments += "--auto-approve-sample"
    }
}
else {
    $vroArguments = @(
        "run", "python", "-m", "vision_research_ops.cli.research",
        "--mode", "live",
        "--output-root", $vroRunRoot,
        "--workflow-id", $vroWorkflowId
    )
}

if ($DryRun) {
    [ordered]@{
        mode = $Mode
        workflow_id = $vroWorkflowId
        run_root = $vroRunRoot
        command = "uv " + ($vroArguments -join " ")
        api_key_present = -not [string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY)
        env_file_present = Test-Path -LiteralPath $vroEnvFile
    } | ConvertTo-Json
    exit 0
}

$vroDisplayName = if ($Mode -eq "agent") { "SEM Research Agent" } else { "SEM Research Agent $Mode" }
Write-Host "Starting $vroDisplayName"
Write-Host "Workflow: $vroWorkflowId"
& uv @vroArguments
$vroExitCode = $LASTEXITCODE
Write-Host "SEM Research Agent exited with code $vroExitCode"
exit $vroExitCode
