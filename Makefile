.PHONY: help build build-nc up up-fg down restart status ps logs test test-docker shell run-local clean

# Default goal
.DEFAULT_GOAL := help

# Deteksi ketersediaan perintah di sistem host
DOCKER_AVAILABLE := $(shell command -v docker 2> /dev/null)
PYTEST_LOCAL     := $(shell command -v pytest 2> /dev/null)

# Warna terminal
CYAN   := \033[36m
GREEN  := \033[32m
YELLOW := \033[33m
RED    := \033[31m
BOLD   := \033[1m
RESET  := \033[0m

help: ## Menampilkan panduan seluruh perintah yang tersedia
	@echo ""
	@echo "$(CYAN)$(BOLD)======================================================$(RESET)"
	@echo "$(CYAN)$(BOLD)       CekTimpa: Copy-Move Forgery Detector CLI       $(RESET)"
	@echo "$(CYAN)$(BOLD)======================================================$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

# ==========================================
# Docker Lifecycle (Responsif & Cepat)
# ==========================================

up: ## Menjalankan aplikasi secara instan di background (port 8501)
	@echo "$(GREEN)Memulai CekTimpa di background...$(RESET)"
	@docker compose up -d
	@echo "$(CYAN)✓ CekTimpa aktif: $(BOLD)http://localhost:8501$(RESET)"
	@echo "$(YELLOW)Gunakan 'make logs' untuk melihat log atau 'make down' untuk menghentikan.$(RESET)"

up-fg: ## Menjalankan aplikasi di foreground (menampilkan log langsung)
	@echo "$(GREEN)Menjalankan CekTimpa di foreground (http://localhost:8501)...$(RESET)"
	docker compose up

down: ## Menghentikan dan membersihkan container Docker
	@echo "$(YELLOW)Menghentikan CekTimpa container...$(RESET)"
	docker compose down
	@echo "$(GREEN)✓ Container berhasil dihentikan.$(RESET)"

restart: down up ## Me-restart container secara cepat

status: ps ## Cek status container dan port aktif
ps:
	@echo "$(CYAN)Status container CekTimpa:$(RESET)"
	@docker compose ps

logs: ## Melihat log real-time dari aplikasi
	docker compose logs -f --tail=100

shell: ## Membuka terminal interaktif (bash) di dalam container
	@echo "$(GREEN)Membuka shell di dalam container CekTimpa...$(RESET)"
	@docker compose exec cektimpa bash || docker compose run --rm cektimpa bash

build: ## Membangun image Docker (menggunakan cache untuk kecepatan)
	@echo "$(YELLOW)Membangun image Docker CekTimpa...$(RESET)"
	docker compose build

build-nc: ## Membangun ulang image Docker dari awal tanpa cache
	@echo "$(RED)Membangun ulang image Docker tanpa cache...$(RESET)"
	docker compose build --no-cache

# ==========================================
# Pengujian (Smart Fallback: Local / Docker)
# ==========================================

test: ## Menjalankan unit tests (otomatis pilih lokal jika ada pytest, atau lewat Docker)
ifdef PYTEST_LOCAL
	@echo "$(GREEN)Menjalankan pytest di host lokal...$(RESET)"
	pytest -v tests/
else
	@echo "$(YELLOW)Pytest lokal tidak ditemukan. Menjalankan via Docker container...$(RESET)"
	docker compose run --rm cektimpa pytest -v tests/
endif

test-docker: ## Menjalankan pengujian unit (pytest) khusus di dalam Docker
	@echo "$(GREEN)Menjalankan pytest di dalam Docker container...$(RESET)"
	docker compose run --rm cektimpa pytest -v tests/

# ==========================================
# Native Python (Tanpa Docker)
# ==========================================

run-local: ## Menjalankan aplikasi Streamlit secara lokal (jika terpasang di host)
	@echo "$(GREEN)Menjalankan Streamlit secara lokal di http://localhost:8501...$(RESET)"
	streamlit run app/main.py

# ==========================================
# Cleanup
# ==========================================

clean: ## Membersihkan cache Python dan file temporer
	@echo "$(YELLOW)Membersihkan cache (__pycache__, .pytest_cache)...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cache bersih!$(RESET)"
