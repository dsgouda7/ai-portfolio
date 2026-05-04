# Batch update all exercises with minimal Makefiles and docker-compose files
# Run from: exercises/ directory

$mlExercises = @(
    @{path="01-ml/03-neural-networks"; name="nn-api"; port="5003"},
    @{path="01-ml/04-recommender-systems"; name="recsys-api"; port="5004"},
    @{path="01-ml/05-anomaly-detection"; name="anomaly-api"; port="5005"},
    @{path="01-ml/06-reinforcement-learning"; name="rl-api"; port="5006"},
    @{path="01-ml/07-unsupervised-learning"; name="unsupervised-api"; port="5007"},
    @{path="01-ml/08-ensemble-methods"; name="ensemble-api"; port="5008"}
)

Write-Host "`n=== Updating ML Exercises ===" -ForegroundColor Cyan

foreach ($ex in $mlExercises) {
    Write-Host "`n→ $($ex.path)..." -ForegroundColor Yellow
    
    $makefilePath = Join-Path $ex.path "Makefile"
    $composePath = Join-Path $ex.path "docker-compose.yml"
    
    # Makefile
    @"
# $($ex.name) - ML Exercise
# Uses shared infrastructure from ../../_infrastructure/

PROJECT_NAME = $($ex.name)
DOCKERFILE = Dockerfile.ml

include ../../_infrastructure/Makefile.include

# Exercise-specific target
train:
	python main.py
"@ | Out-File $makefilePath -Encoding UTF8 -NoNewline
    
    # docker-compose.yml
    @"
version: '3.8'

services:
  $($ex.name):
    build:
      context: .
      dockerfile: ../../_infrastructure/docker/Dockerfile.ml
    container_name: $($ex.name)
    ports:
      - "$($ex.port):5000"
    volumes:
      - ./src:/app/src:ro
      - ./config.yaml:/app/config.yaml:ro
      - ./models:/app/models:ro
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - PYTHONUNBUFFERED=1
    networks:
      - ml-network
    restart: unless-stopped

networks:
  ml-network:
    external: true
"@ | Out-File $composePath -Encoding UTF8
    
    Write-Host "  ✓ Makefile + docker-compose updated" -ForegroundColor Green
}

# Grand Challenges
$grandChallenges = @(
    @{path="02-advanced_deep_learning"; name="adv-dl-api"; port="5102"; dockerfile="Dockerfile.dl"; infra="../_infrastructure"},
    @{path="03-ai"; name="ai-api"; port="5103"; dockerfile="Dockerfile.api"; infra="../_infrastructure"},
    @{path="04-multi_agent_ai"; name="multi-agent-api"; port="5104"; dockerfile="Dockerfile.api"; infra="../_infrastructure"},
    @{path="05-multimodal_ai"; name="multimodal-api"; port="5105"; dockerfile="Dockerfile.dl"; infra="../_infrastructure"},
    @{path="06-ai_infrastructure"; name="ai-infra-api"; port="5106"; dockerfile="Dockerfile.api"; infra="../_infrastructure"}
)

Write-Host "`n=== Updating Grand Challenge Exercises ===" -ForegroundColor Cyan

foreach ($ex in $grandChallenges) {
    Write-Host "`n→ $($ex.path)..." -ForegroundColor Yellow
    
    $makefilePath = Join-Path $ex.path "Makefile"
    $composePath = Join-Path $ex.path "docker-compose.yml"
    
    # Makefile
    @"
# $($ex.name) - Grand Challenge
# Uses shared infrastructure from ../_infrastructure/

PROJECT_NAME = $($ex.name)
DOCKERFILE = $($ex.dockerfile)

include ../_infrastructure/Makefile.include

# Exercise-specific target
train:
	python main.py
"@ | Out-File $makefilePath -Encoding UTF8 -NoNewline
    
    # docker-compose.yml
    @"
version: '3.8'

services:
  $($ex.name):
    build:
      context: .
      dockerfile: ../_infrastructure/docker/$($ex.dockerfile)
    container_name: $($ex.name)
    ports:
      - "$($ex.port):5000"
    volumes:
      - ./src:/app/src:ro
      - ./config.yaml:/app/config.yaml:ro
      - ./models:/app/models:ro
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - PYTHONUNBUFFERED=1
    networks:
      - ml-network
    restart: unless-stopped

networks:
  ml-network:
    external: true
"@ | Out-File $composePath -Encoding UTF8
    
    Write-Host "  ✓ Makefile + docker-compose updated" -ForegroundColor Green
}

Write-Host "`n✅ All exercises updated successfully!" -ForegroundColor Green
Write-Host "Total exercises processed: $($mlExercises.Count + $grandChallenges.Count)" -ForegroundColor Cyan
