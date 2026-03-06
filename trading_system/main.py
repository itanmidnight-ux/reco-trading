from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_system.app.main import api, launch_services

__all__ = ['api', 'launch_services']

if __name__ == '__main__':
    if os.getenv('RUNTIME_PROFILE', '').strip().lower() == 'production':
        raise RuntimeError(
            'trading_system/main.py is development_only and not a production runtime entrypoint. '
            'Use run.sh -> main.py -> QuantKernel.run().'
        )
    import asyncio
    import uvicorn

    async def _main() -> None:
        task = asyncio.create_task(launch_services())
        config = uvicorn.Config(api, host='0.0.0.0', port=8000, log_level='info')
        server = uvicorn.Server(config)
        await asyncio.gather(task, server.serve())

    asyncio.run(_main())
