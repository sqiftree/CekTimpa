.PHONY: help build up down restart logs test test-docker run-local clean

# Default goal
.DEFAULT_GOAL := help

# Warna terminal
BLUE   := \033[36m
GREEN  := \033[32m
YELLOW := \033[33m
RESET  := \033[0m

help: ## Menampilkan panduan seluruh perintah yang tersedia
	@echo ""
	@echo "$(BLUE)======================================================$(RESET)"
	@echo "$(BLUE)       CekTimpa: Copy-Move Forgery Detector CLI       $(RESET)"
	@echo "$(BLUE)======================================================$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

# ==========================================
# Docker & WSL Commands
# ==========================================

build: ## Membangun ulang (build) image Docker
	@echo "$(YELLOW)Building Docker image...$(RESET)"
	docker compose build

up: ## Menjalankan aplikasi di dalam Docker (live-reload aktif di port 8501)
	@echo "$(GREEN)Starting CekTimpa on Docker (http://localhost:8501)...$(RESET)"
	docker compose up --build

down: ## Menghentikan dan membersihkan container Docker
	@echo "$(YELLOW)Stopping CekTimpa container...$(RESET)"
	docker compose down

restart: down up ## Me-restart container Docker

logs: ## Melihat log real-time dari container Docker
	docker compose logs -f

test-docker: ## Menjalankan pengujian unit (pytest) di dalam Docker
	@echo "$(GREEN)Running pytest inside Docker container...$(RESET)"
	docker compose run --rm cektimpa pytest -v tests/

# ==========================================
# Native Python / Local Commands
# ==========================================

run-local: ## Menjalankan aplikasi Streamlit secara lokal (tanpa Docker)
	@echo "$(GREEN)Starting Streamlit locally on http://localhost:8501...$(RESET)"
	streamlit run app/main.py

test: ## Menjalankan pengujian unit (pytest) di lingkungan Python lokal
	@echo "$(GREEN)Running pytest locally...$(RESET)"
	pytest -v tests/

# ==========================================
# Cleanup Commands
# ==========================================

clean: ## Membersihkan cache Python dan file temporer
	@echo "$(YELLOW)Cleaning python cache and artifacts...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Done!$(RESET)"
