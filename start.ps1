# Start IT RAG Chatbot (Backend + Frontend)
Write-Host "Starting IT RAG Chatbot..." -ForegroundColor Cyan

# Start FastAPI backend
$backend = Start-Process -FilePath "python" `
  -ArgumentList "-m uvicorn backend.main:app --reload --port 8000" `
  -WorkingDirectory $PSScriptRoot `
  -PassThru -WindowStyle Normal

Write-Host "Backend started (PID $($backend.Id)) → http://localhost:8000" -ForegroundColor Green
Write-Host "Swagger UI: http://localhost:8000/docs" -ForegroundColor Green

# Start React frontend
$frontend = Start-Process -FilePath "cmd" `
  -ArgumentList "/c npm run dev" `
  -WorkingDirectory "$PSScriptRoot\frontend" `
  -PassThru -WindowStyle Normal

Start-Sleep -Seconds 3
Write-Host "Frontend started (PID $($frontend.Id)) → http://localhost:5173" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to stop both servers..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

$backend | Stop-Process -Force -ErrorAction SilentlyContinue
$frontend | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "Servers stopped." -ForegroundColor Red
