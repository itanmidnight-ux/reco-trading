# Deploy (systemd + Linux tuning + Docker Compose opcional)

## 1) Unidades systemd

Copiar unidades y habilitar:

```bash
sudo cp deploy/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now orchestrator trading-worker evolution-worker architecture-search-worker self-healing-worker
```

Las unidades ejecutan `deploy/linux/run_service.sh` y permiten sobreescribir comandos por rol usando variables de entorno:
- `ORCHESTRATOR_CMD`
- `TRADING_WORKER_CMD`
- `EVOLUTION_WORKER_CMD`
- `ARCHITECTURE_SEARCH_WORKER_CMD`
- `SELF_HEALING_WORKER_CMD`

## 2) Tuning Linux

### ulimit

```bash
deploy/linux/tune_ulimit.sh trading
```

### Networking (kernel 6.x)

```bash
deploy/linux/tune_kernel_network.sh
```

### Afinidad CPU opcional

```bash
deploy/linux/set_cpu_affinity.sh orchestrator.service 0-3
```

### Aislamiento por cgroups

```bash
deploy/linux/setup_cgroup_isolation.sh orchestrator.service 300% 8G 800
```

## 3) Perfiles de despliegue

### Bare-metal (recomendado)

```bash
cp deploy/profiles/bare-metal-systemd.env .env
# editar secretos y endpoints
```

### Docker Compose opcional

```bash
docker compose -f deploy/profiles/docker-compose.optional.yml --profile infra --profile core up -d
# o levantar workers
# docker compose -f deploy/profiles/docker-compose.optional.yml --profile workers up -d
```

## 4) Validaciones de compatibilidad

```bash
deploy/linux/validate_compatibility.sh
```

Valida:
- Python 3.11+
- CUDA (si `nvidia-smi` está disponible)
- Redis (`redis-cli ping`)
- PostgreSQL (`pg_isready`)
