$ErrorActionPreference = 'Stop'
$portalPath = (& wsl.exe -d Ubuntu wslpath -u $PSScriptRoot).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Não foi possível localizar o projeto no WSL Ubuntu.' }
Write-Host 'Abra http://127.0.0.1:7860 após a inicialização.'
& wsl.exe -d Ubuntu --cd $portalPath bash ./iniciar_portal.sh
