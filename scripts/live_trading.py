from __future__ import annotations

import asyncio

from reco_trading.kernel.quant_kernel import QuantKernel


if __name__ == '__main__':
    asyncio.run(QuantKernel().run())
