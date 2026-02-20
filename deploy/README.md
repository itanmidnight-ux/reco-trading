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


## 5) Servicio dedicado `quant-kernel`

Se incluye un ejemplo de unidad dedicado en `deploy/systemd/quant-kernel.service`.

```bash
sudo cp deploy/systemd/quant-kernel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now quant-kernel
```

## 6) Entorno virtual limpio (recomendado)

```bash
cd /opt/reco-trading
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
cp .env.example .env
# editar secretos
```

Validar que el runtime quedó saludable:

```bash
python -m reco_trading.kernel.quant_kernel
```

## 7) Recomendaciones Linux 6.x en Debian 13

- Mantener kernel `6.x` estable de Debian 13 (`linux-image-amd64`) para eBPF, scheduler y mejoras de red.
- Aplicar `deploy/linux/tune_kernel_network.sh` para buffers TCP/UDP y backlog.
- Configurar `LimitNOFILE=65535` (o mayor) vía systemd y también `/etc/security/limits.d/`.
- Activar `irqbalance` y fijar afinidad CPU para servicios críticos (`deploy/linux/set_cpu_affinity.sh`).
- Verificar THP, swap y presión de memoria antes de sesión live (`vm.swappiness`, `vm.max_map_count`).
- Para hardware NUMA, alinear afinidad de workers y Redis/PostgreSQL al mismo nodo NUMA para reducir latencia.
