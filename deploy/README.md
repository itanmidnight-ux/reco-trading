# Deploy (systemd + Linux tuning + Docker Compose opcional)

## 1) Recomendaciones PostgreSQL/Redis para latencia estable

### PostgreSQL (OLTP de baja latencia)

- Usa NVMe local y evita NFS para `PGDATA`.
- Mantén `shared_buffers` en ~25% de RAM y `effective_cache_size` en 50-70%.
- Activa `synchronous_commit=off` **solo** para tablas/eventos no críticos de telemetría.
- Fija `max_connections` bajo (ej. 100) y usa pool async en la app.
- Ajustes sugeridos en `postgresql.conf`:

```conf
shared_buffers = 4GB
effective_cache_size = 12GB
wal_compression = on
checkpoint_timeout = 15min
max_wal_size = 4GB
random_page_cost = 1.1
```

### Redis (estado caliente)

- `appendonly yes` + `appendfsync everysec` para balancear durabilidad/latencia.
- Desactiva THP en Linux y fija `vm.overcommit_memory=1`.
- Reserva memoria con `maxmemory` y política `allkeys-lru` para evitar OOM.
- En producción usa instancia dedicada para el kernel (sin mezclar con colas externas).

## 2) Unidades systemd

Copiar unidades y habilitar:

```bash
sudo cp deploy/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now quant-kernel orchestrator trading-worker evolution-worker
```

> `quant-kernel.service` levanta `python main.py` como proceso principal de ejecución.

Las unidades ejecutan `deploy/linux/run_service.sh` y permiten sobreescribir comandos por rol usando variables de entorno:
- `ORCHESTRATOR_CMD`
- `TRADING_WORKER_CMD`
- `EVOLUTION_WORKER_CMD`
- `ARCHITECTURE_SEARCH_WORKER_CMD` (perfil `research`)
- `SELF_HEALING_WORKER_CMD` (perfil `research`)

## 3) Tuning Linux

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
deploy/linux/set_cpu_affinity.sh quant-kernel.service 0-3
```

### Aislamiento por cgroups

```bash
deploy/linux/setup_cgroup_isolation.sh quant-kernel.service 300% 8G 800
```

## 4) Perfiles de despliegue

### Bare-metal (recomendado)

```bash
cp deploy/profiles/bare-metal-systemd.env .env
# editar secretos y endpoints
```

### Docker Compose opcional

```bash
docker compose -f deploy/profiles/docker-compose.optional.yml --profile infra --profile core up -d
# workers productivos
# docker compose -f deploy/profiles/docker-compose.optional.yml --profile workers up -d
# componentes experimentales/no conectados
# docker compose -f deploy/profiles/docker-compose.optional.yml --profile research up -d
```

## 5) Guía de arranque y recuperación

### Arranque limpio

1. Validar runtime:
   ```bash
   deploy/linux/validate_compatibility.sh
   ```
2. Verificar conectividad a Redis/PostgreSQL.
3. Aplicar esquema SQL (`database/schema.sql`) y variables `.env.template`.
4. Iniciar `quant-kernel`:
   ```bash
   sudo systemctl start quant-kernel
   sudo systemctl status quant-kernel --no-pager
   ```

### Recuperación operativa (incidente)

1. Congelar ejecución:
   ```bash
   sudo systemctl stop quant-kernel
   ```
2. Revisar logs recientes:
   ```bash
   journalctl -u quant-kernel -n 200 --no-pager
   ```
3. Verificar salud de infraestructura (`pg_isready`, `redis-cli ping`).
4. Si hay corrupción de estado de sesión, limpia únicamente claves del namespace del kernel en Redis.
5. Reiniciar en modo conservador (`CONSERVATIVE_MODE_ENABLED=true`) y levantar:
   ```bash
   sudo systemctl restart quant-kernel
   ```

## 6) Validaciones de compatibilidad

```bash
deploy/linux/validate_compatibility.sh
```

Valida:
- Python 3.11+
- CUDA (si `nvidia-smi` está disponible)
- Redis (`redis-cli ping`)
- PostgreSQL (`pg_isready`)
