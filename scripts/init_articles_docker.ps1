# PowerShell script to initialize articles in Docker environment

Write-Host "📰 Initializing articles in Docker environment..." -ForegroundColor Cyan

# Check if services are running
try {
    $healthResponse = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 5
    Write-Host "✅ Services are running" -ForegroundColor Green
} catch {
    Write-Host "❌ Services are not running. Please run .\scripts\run_docker.ps1 first" -ForegroundColor Red
    exit 1
}

# List of article URLs to initialize
$articleUrls = @(
    "https://www.bbc.com/news/technology-67833126",
    "https://www.theguardian.com/technology/2024/jan/15/ai-tools-reshape-workplace",
    "https://www.reuters.com/technology/artificial-intelligence/",
    "https://techcrunch.com/2024/01/10/the-future-of-ai-in-2024/",
    "https://www.wired.com/story/artificial-intelligence-climate-change/",
    "https://www.nature.com/articles/d41586-024-00001-9",
    "https://arxiv.org/abs/2401.12345",
    "https://blog.openai.com/new-models-and-developer-products-announced/",
    "https://ai.googleblog.com/2024/01/advancing-ai-research.html",
    "https://www.microsoft.com/en-us/research/blog/ai-breakthrough-2024/",
    "https://www.deepmind.com/blog/advancing-science-with-ai",
    "https://www.anthropic.com/news/constitutional-ai-progress",
    "https://www.nvidia.com/en-us/ai-data-science/",
    "https://www.ibm.com/watson/ai-ethics",
    "https://aws.amazon.com/machine-learning/",
    "https://cloud.google.com/ai-platform",
    "https://azure.microsoft.com/en-us/products/machine-learning/",
    "https://www.tesla.com/AI",
    "https://www.spacex.com/updates/starship-ai-systems.html",
    "https://www.mit.edu/~amini/artificial-intelligence-for-social-good/"
)

Write-Host "🔄 Adding $($articleUrls.Count) articles..." -ForegroundColor Yellow

$successCount = 0
$failureCount = 0

foreach ($url in $articleUrls) {
    try {
        Write-Host "📄 Adding: $url" -ForegroundColor White
        
        $requestBody = @{
            url = $url
        } | ConvertTo-Json -Depth 10
        
        $response = Invoke-RestMethod -Uri "http://localhost:8000/ingest" -Method Post -Body $requestBody -ContentType "application/json" -TimeoutSec 30
        
        if ($response.success) {
            Write-Host "   ✅ Success: $($response.message)" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host "   ⚠️  Warning: $($response.message)" -ForegroundColor Yellow
            $failureCount++
        }
    } catch {
        Write-Host "   ❌ Failed: $($_.Exception.Message)" -ForegroundColor Red
        $failureCount++
    }
    
    # Small delay to avoid overwhelming the API
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "📊 Article initialization complete!" -ForegroundColor Cyan
Write-Host "   ✅ Successful: $successCount" -ForegroundColor Green
Write-Host "   ❌ Failed: $failureCount" -ForegroundColor Red

# Check final status
try {
    $finalHealth = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
    Write-Host "   📚 Total articles in database: $($finalHealth.articles_count)" -ForegroundColor Cyan
} catch {
    Write-Host "   ⚠️  Could not check final article count" -ForegroundColor Yellow
}

if ($successCount -gt 0) {
    Write-Host ""
    Write-Host "🎉 Articles are ready! You can now:" -ForegroundColor Green
    Write-Host "   🌐 Use the web UI: http://localhost:8000" -ForegroundColor White
    Write-Host "   🔍 Try asking questions about the articles" -ForegroundColor White
}