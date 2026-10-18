#requires -Version 5.1
<#
.SYNOPSIS
    Publish the already-running Forte API through a temporary Cloudflare HTTPS URL.
.DESCRIPTION
    CheckOnly performs local checks without creating a tunnel. A real launch requires
    explicit AllowExternalTraffic after approval of the external data route.
    Run in a terminal and keep it open; Ctrl+C stops only the child created here.
    The inference server at 127.0.0.1:8777 is never a permitted origin.
#>
[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8000,
    [string]$CloudflaredPath = '',
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ExpectedSha256 = '4d2490e74082c5d48dbfedaf92576fa5bd1702379ab88eb7faa0cfbe8dc891b6',
    [ValidateRange(10, 300)]
    [int]$StartupTimeoutSeconds = 90,
    [switch]$CheckOnly,
    [switch]$AllowExternalTraffic
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$tunnelRoot = Join-Path $repositoryRoot 'outputs/forte-tunnel'
$origin = "http://127.0.0.1:$Port"

function Get-LoopbackResponse {
    param([string]$Url)
    # Do not use an inherited HTTP proxy, cookies, credentials, or redirects.
    $request = [Net.HttpWebRequest]::Create($Url)
    $request.Proxy = $null
    $request.AllowAutoRedirect = $false
    $request.UseDefaultCredentials = $false
    $request.Timeout = 5000
    $request.ReadWriteTimeout = 5000
    $response = $null
    $reader = $null
    try {
        try {
            $response = $request.GetResponse()
        }
        catch [Net.WebException] {
            if ($null -eq $_.Exception.Response) {
                throw 'Local API is unreachable; start the Forte service first.'
            }
            $response = $_.Exception.Response
        }
        $reader = New-Object IO.StreamReader($response.GetResponseStream())
        $buffer = New-Object char[] 8192
        $count = $reader.ReadBlock($buffer, 0, $buffer.Length)
        if ($count -eq $buffer.Length) {
            throw 'Unexpectedly large response from the local API.'
        }
        return [PSCustomObject]@{
            Status = [int]$response.StatusCode
            Body = [string]::new($buffer, 0, $count)
        }
    }
    finally {
        if ($null -ne $reader) { $reader.Dispose() }
        if ($null -ne $response) { $response.Dispose() }
    }
}

function Write-ConnectionState {
    param([string]$Path, [System.Collections.IDictionary]$State)
    $temporary = $Path + '.tmp'
    [IO.File]::WriteAllText($temporary, ($State | ConvertTo-Json -Depth 4),
        (New-Object Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Read-SharedTunnelLog {
    param([string]$Path)
    $stream = $null
    $memory = $null
    $reader = $null
    try {
        # Go keeps the log open for writing. The reader must allow that writer
        # and log rotation, not just share read access as ReadAllText does.
        $sharing = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
        $stream = [IO.FileStream]::new($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, $sharing)
        # Bound bytes before UTF-8 decoding, even if a writer grows the file.
        $buffer = New-Object byte[] (1024 * 1024)
        $count = 0
        while ($count -lt $buffer.Length) {
            $read = $stream.Read($buffer, $count, $buffer.Length - $count)
            if ($read -eq 0) { break }
            $count += $read
        }
        $memory = [IO.MemoryStream]::new($buffer, 0, $count, $false)
        $reader = [IO.StreamReader]::new($memory, [Text.Encoding]::UTF8, $true)
        return $reader.ReadToEnd()
    }
    catch [IO.FileNotFoundException] { return $null }
    catch [IO.DirectoryNotFoundException] { return $null }
    finally {
        if ($null -ne $reader) { $reader.Dispose() }
        if ($null -ne $memory) { $memory.Dispose() }
        if ($null -ne $stream) { $stream.Dispose() }
    }
}

if ($env:OS -ne 'Windows_NT') {
    throw 'This launcher is for Windows; the prepared binary is Windows amd64.'
}
if ($Port -eq 8777) {
    throw 'Port 8777 is the private model server and must not be exposed.'
}
if (-not $CheckOnly -and -not $AllowExternalTraffic) {
    throw 'External route is not approved. Use -CheckOnly, or -AllowExternalTraffic only after permission.'
}
if ([string]::IsNullOrWhiteSpace($CloudflaredPath)) {
    $CloudflaredPath = Join-Path $tunnelRoot 'tools/cloudflared-windows-amd64.exe'
}
$binary = (Resolve-Path -LiteralPath $CloudflaredPath).ProviderPath
if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
    throw 'cloudflared must be an existing local executable.'
}
$binaryStream = [IO.File]::OpenRead($binary)
$hashAlgorithm = [Security.Cryptography.SHA256]::Create()
try {
    $actualHash = [BitConverter]::ToString($hashAlgorithm.ComputeHash($binaryStream)).Replace('-', '')
}
finally {
    $binaryStream.Dispose()
    $hashAlgorithm.Dispose()
}
if ($actualHash -ne $ExpectedSha256) {
    throw 'cloudflared SHA-256 mismatch. Do not execute this file; verify the official release.'
}

$health = Get-LoopbackResponse ($origin + '/healthz')
if ($health.Status -ne 200) { throw 'Local health check did not return HTTP 200.' }
$healthBody = $health.Body | ConvertFrom-Json
if ($healthBody.status -ne 'ok' -or $healthBody.model_ready -isnot [bool] -or
    -not $healthBody.model_ready) {
    throw 'The local model is not ready; no tunnel was created.'
}
$authentication = Get-LoopbackResponse ($origin + '/api/v1/jobs/tunnel-preflight-no-key')
if ($authentication.Status -ne 401) {
    throw 'API must reject requests without X-API-Key (HTTP 401); no tunnel was created.'
}
if ($CheckOnly) {
    [PSCustomObject]@{
        status = 'LOCAL_CHECKS_PASSED_NO_TUNNEL'
        origin = $origin
        model_ready = $true
        unauthenticated_api_status = 401
        binary_sha256 = $actualHash.ToLowerInvariant()
    } | ConvertTo-Json -Compress
    return
}

[IO.Directory]::CreateDirectory($tunnelRoot) | Out-Null
$lockPath = Join-Path $tunnelRoot 'tunnel.lock'
$lock = $null
$ownedProcess = $null
$processStarted = $false
$runDirectory = $null
$statePath = $null
$state = $null
try {
    try {
        $lock = [IO.FileStream]::new($lockPath, [IO.FileMode]::CreateNew,
            [IO.FileAccess]::ReadWrite, [IO.FileShare]::None, 4096, [IO.FileOptions]::DeleteOnClose)
    }
    catch [IO.IOException] {
        throw 'A tunnel lock exists. Inspect its owner; this launcher never removes another lock.'
    }
    $lockBytes = [Text.Encoding]::UTF8.GetBytes("launcher_pid=$PID`n")
    $lock.Write($lockBytes, 0, $lockBytes.Length)
    $lock.Flush()
    $runId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [Guid]::NewGuid().ToString('N')
    $runDirectory = Join-Path $tunnelRoot ('runs/' + $runId)
    $logDirectory = Join-Path $runDirectory 'logs'
    [IO.Directory]::CreateDirectory($logDirectory) | Out-Null
    $configPath = Join-Path $runDirectory 'empty-config.yml'
    [IO.File]::WriteAllText($configPath, "{}`n", (New-Object Text.UTF8Encoding($false)))
    $statePath = Join-Path $runDirectory 'connection.json'
    $arguments = @('tunnel', '--config', $configPath, '--url', $origin, '--protocol', 'http2',
        '--no-autoupdate', '--loglevel', 'info', '--log-directory', $logDirectory,
        '--metrics', '127.0.0.1:0')
    # Windows filenames cannot contain a quote. These generated arguments have
    # no trailing slash, so explicit quotes also work under Windows PowerShell 5.1.
    foreach ($argument in $arguments) {
        if ($argument.Contains('"') -or $argument.Contains("`n") -or $argument.EndsWith('\')) {
            throw 'Unsupported command argument; cloudflared was not started.'
        }
    }
    $startInfo = New-Object Diagnostics.ProcessStartInfo
    $startInfo.FileName = $binary
    $startInfo.Arguments = ($arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $startInfo.WorkingDirectory = $runDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($name in @($startInfo.EnvironmentVariables.Keys)) {
        if ($name -match '^(FORTE_|TUNNEL_|CLOUDFLARE_|CF_)' -or $name -match '_PROXY$') {
            $startInfo.EnvironmentVariables.Remove($name)
        }
    }
    $ownedProcess = New-Object Diagnostics.Process
    $ownedProcess.StartInfo = $startInfo
    if (-not $ownedProcess.Start()) { throw 'Could not start the owned cloudflared process.' }
    $processStarted = $true
    # Retain the process handle; never look up a process by name or a stale PID.
    $null = $ownedProcess.Handle
    $stdoutDrain = $ownedProcess.StandardOutput.BaseStream.CopyToAsync([IO.Stream]::Null)
    $stderrDrain = $ownedProcess.StandardError.BaseStream.CopyToAsync([IO.Stream]::Null)
    $state = [ordered]@{
        status = 'starting'
        launcher_pid = $PID
        tunnel_pid = $ownedProcess.Id
        tunnel_started_utc = $ownedProcess.StartTime.ToUniversalTime().ToString('o')
        origin = $origin
        public_url = $null
        binary_sha256 = $actualHash.ToLowerInvariant()
    }
    Write-ConnectionState $statePath $state
    $startup = [Diagnostics.Stopwatch]::StartNew()
    $publicUrl = $null
    while (-not $ownedProcess.HasExited) {
        if ($null -eq $publicUrl) {
            $logPath = Join-Path $logDirectory 'cloudflared.log'
            $logText = Read-SharedTunnelLog $logPath
            if ($null -ne $logText) {
                $match = [regex]::Match($logText, 'https://[a-z0-9-]+\.trycloudflare\.com\b')
                if ($match.Success) {
                    $publicUrl = $match.Value
                    $state.status = 'url_issued_external_reachability_not_verified'
                    $state.public_url = $publicUrl
                    Write-ConnectionState $statePath $state
                    Write-Output ("Temporary HTTPS URL: " + $publicUrl)
                    Write-Output 'Keep this terminal and the Forte service running. Ctrl+C stops this tunnel.'
                    Write-Output 'The URL is issued; verify reachability from a different network before submission.'
                    Write-Output ("Connection metadata: " + $statePath)
                }
            }
            if ($null -eq $publicUrl -and $startup.Elapsed.TotalSeconds -ge $StartupTimeoutSeconds) {
                throw ('No tunnel URL before deadline. Inspect the local logs in ' + $logDirectory)
            }
        }
        Start-Sleep -Milliseconds 500
        $ownedProcess.Refresh()
    }
    throw ('Owned cloudflared exited. Inspect the local logs in ' + $logDirectory)
}
finally {
    try {
        if ($null -ne $ownedProcess) {
            try {
                if ($processStarted -and -not $ownedProcess.HasExited) {
                    $ownedProcess.Kill()
                    $null = $ownedProcess.WaitForExit(5000)
                }
            }
            finally {
                $ownedProcess.Dispose()
            }
        }
    }
    finally {
        try {
            if ($null -ne $statePath -and $null -ne $state) {
                $state.status = 'stopped'
                $state.stopped_utc = [DateTime]::UtcNow.ToString('o')
                Write-ConnectionState $statePath $state
            }
        }
        finally {
            if ($null -ne $lock) { $lock.Dispose() }
        }
    }
}
