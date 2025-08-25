# PowerShell script to run the Article Chat System with Docker

Write-Host "🐳 Starting Article Chat System with Docker..." -ForegroundColor Cyan

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  .env file not found. Creating template..." -ForegroundColor Yellow
    @"
GOOGLE_API_KEY=your_google_api_key_here
"@ | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "📝 Please edit .env file and add your Google API key" -ForegroundColor Yellow
    Write-Host "   You can get one from: https://console.cloud.google.com/" -ForegroundColor Yellow
    exit 1
}

# Load environment variables
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^([^#][^=]*?)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
        }
    }
}

# Check if GOOGLE_API_KEY is set
if (-not $env:GOOGLE_API_KEY -or $env:GOOGLE_API_KEY -eq "your_google_api_key_here") {
    Write-Host "❌ GOOGLE_API_KEY not set in .env file" -ForegroundColor Red
    exit 1
}

Write-Host "🏗️  Building and starting services..." -ForegroundColor Cyan

# Stop existing containers if running
docker-compose down --remove-orphans

# Build and start services
docker-compose up --build -d

# Wait for services to be healthy
Write-Host "⏳ Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep 10

# Check service health
$chromaHealthy = $false
$appHealthy = $false

for ($i = 1; $i -le 30; $i++) {
    try {
        # Check ChromaDB
        $chromaResponse = Invoke-RestMethod -Uri "http://localhost:8001/api/v1/heartbeat" -Method Get -TimeoutSec 5
        if ($chromaResponse) {
            $chromaHealthy = $true
            Write-Host "✅ ChromaDB is healthy" -ForegroundColor Green
        }
    } catch {
        # ChromaDB not ready yet
    }
    
    try {
        # Check main app
        $appResponse = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 5
        if ($appResponse.status -eq "healthy") {
            $appHealthy = $true
            Write-Host "✅ Article Chat API is healthy" -ForegroundColor Green
        }
    } catch {
        # App not ready yet
    }
    
    if ($chromaHealthy -and $appHealthy) {
        break
    }
    
    Write-Host "⏳ Services starting... ($i/30)" -ForegroundColor Yellow
    Start-Sleep 2
}

if (-not $chromaHealthy) {
    Write-Host "❌ ChromaDB failed to start" -ForegroundColor Red
    docker-compose logs chroma-db
    exit 1
}

if (-not $appHealthy) {
    Write-Host "❌ Article Chat API failed to start" -ForegroundColor Red
    docker-compose logs article-chat
    exit 1
}

Write-Host "" 
Write-Host "🎉 All services are running successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Service URLs:" -ForegroundColor Cyan
Write-Host "   🌐 Web UI:        http://localhost:8000" -ForegroundColor White
Write-Host "   📚 API Docs:      http://localhost:8000/docs" -ForegroundColor White
Write-Host "   🔍 ChromaDB:      http://localhost:8001" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Useful commands:" -ForegroundColor Cyan
Write-Host "   📋 View logs:     docker-compose logs -f" -ForegroundColor White
Write-Host "   🛑 Stop services: docker-compose down" -ForegroundColor White
Write-Host "   🔄 Restart:       docker-compose restart" -ForegroundColor White
Write-Host ""

# Check if articles are loaded
try {
    $healthResponse = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
    if ($healthResponse.articles_count -gt 0) {
        Write-Host "📰 Articles loaded: $($healthResponse.articles_count)" -ForegroundColor Green
    } else {
        Write-Host "⚠️  No articles loaded yet. You can add articles through the web UI." -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Could not check article count" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 Ready to chat about articles! Open http://localhost:8000 in your browser." -ForegroundColor Green