from __future__ import annotations

import asyncio

from reco_trading.kernel import QuantKernel


async def main() -> None:
    kernel = QuantKernel()
    await kernel.initialize()
    await kernel.run()


if __name__ == '__main__':
    asyncio.run(main())
