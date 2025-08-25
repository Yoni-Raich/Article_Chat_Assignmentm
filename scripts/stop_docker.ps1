# PowerShell script to stop the Article Chat System Docker containers

Write-Host "🛑 Stopping Article Chat System..." -ForegroundColor Cyan

# Stop and remove containers
docker-compose down --remove-orphans

Write-Host "✅ All services stopped" -ForegroundColor Green

# Option to remove volumes (data)
$removeData = Read-Host "Do you want to remove the database data as well? (y/N)"
if ($removeData -eq "y" -or $removeData -eq "Y") {
    Write-Host "🗑️  Removing database volumes..." -ForegroundColor Yellow
    docker-compose down --volumes
    docker volume prune -f
    Write-Host "✅ Database data removed" -ForegroundColor Green
} else {
    Write-Host "💾 Database data preserved for next run" -ForegroundColor Green
}

Write-Host ""
Write-Host "🔧 To start again, run: .\scripts\run_docker.ps1" -ForegroundColor Cyan